# -*- coding: utf-8 -*-
"""
Medical OCR Tool using RapidOCR with GPU support.
This tool extracts text from medical image files and performs layout analysis
to identify and extract figure regions from medical reports.
"""
import os
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime
import re

# Try to import cv2 (optional for some OCR backends)
CV2_AVAILABLE = False
cv2 = None
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV (cv2) is not installed. Some image processing features may be limited.")

from agentscope.message import TextBlock
from agentscope.tool._response import ToolResponse

# Try to import RapidOCR
RAPIDOCR_AVAILABLE = False
RapidOCR = None

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    try:
        from rapidocr import RapidOCR
        RAPIDOCR_AVAILABLE = True
    except ImportError:
        print("Warning: RapidOCR is not installed. Please install it with: pip install rapidocr-onnxruntime")

# Check for GPU support in ONNX Runtime
GPU_AVAILABLE = False
try:
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    GPU_AVAILABLE = 'CUDAExecutionProvider' in available_providers or 'TensorrtExecutionProvider' in available_providers
    if GPU_AVAILABLE:
        print(f"OCR GPU 加速可用: {[p for p in available_providers if 'CUDA' in p or 'Tensorrt' in p]}")
except ImportError:
    pass

# Global OCR instance (cached for performance)
_ocr_instance: Optional[Any] = None


def get_ocr_instance() -> Optional[Any]:
    """Get or create a cached RapidOCR instance with GPU support."""
    global _ocr_instance
    if _ocr_instance is None and RAPIDOCR_AVAILABLE:
        try:
            if GPU_AVAILABLE:
                # 使用 GPU 加速
                _ocr_instance = RapidOCR(
                    det_use_cuda=True,
                    rec_use_cuda=True,
                    cls_use_cuda=True
                )
                print("RapidOCR 已启用 GPU 加速")
            else:
                # CPU 模式
                _ocr_instance = RapidOCR()
                print("RapidOCR 使用 CPU 模式")
        except TypeError:
            # 如果不支持 CUDA 参数，回退到默认配置
            try:
                _ocr_instance = RapidOCR()
                print("RapidOCR 初始化成功 (默认配置)")
            except Exception as e:
                print(f"Warning: Failed to initialize RapidOCR: {e}")
                return None
        except Exception as e:
            print(f"Warning: Failed to initialize RapidOCR with GPU: {e}")
            try:
                _ocr_instance = RapidOCR()
                print("RapidOCR 回退到 CPU 模式")
            except Exception as e2:
                print(f"Warning: Failed to initialize RapidOCR: {e2}")
                return None
    return _ocr_instance


def is_figure_path(file_path: str) -> bool:
    """
    Check if the file path is from the figure directory.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if the file is from figure directory
    """
    # 检查路径中是否包含 /figure/ 或 \\figure\\
    normalized_path = file_path.replace('\\', '/')
    return '/data/figure/' in normalized_path or '/figure/' in normalized_path


def extract_text_from_image(
    image_path: str,
    use_angle_cls: bool = True,
    lang: str = 'ch',
) -> ToolResponse:
    """
    Extract text from a medical image using RapidOCR.
    
    Args:
        image_path (str): Path to the image file.
        use_angle_cls (bool): Whether to use angle classifier for text detection.
        lang (str): Language code for OCR. 'ch' for Chinese, 'en' for English.
    
    Returns:
        ToolResponse: A tool response containing the extracted text.
                     Returns error message if OCR fails or RapidOCR is not available.
    """
    if not RAPIDOCR_AVAILABLE:
        return ToolResponse(
            content=[TextBlock(
                type="text",
                text="Error: RapidOCR is not installed. Please install it with: pip install rapidocr-onnxruntime"
            )]
        )
    
    # Check if image file exists
    if not os.path.exists(image_path):
        return ToolResponse(
            content=[TextBlock(
                type="text",
                text=f"Error: Image file not found: {image_path}"
            )]
        )
    
    if not os.path.isfile(image_path):
        return ToolResponse(
            content=[TextBlock(
                type="text",
                text=f"Error: Path is not a file: {image_path}"
            )]
        )
    
    try:
        # Get cached OCR instance
        ocr = get_ocr_instance()
        if ocr is None:
            return ToolResponse(
                content=[TextBlock(
                    type="text",
                    text="Error: Failed to initialize RapidOCR"
                )]
            )
        
        # Perform OCR
        ocr_result = ocr(image_path)
        
        # Handle different return formats
        if isinstance(ocr_result, tuple) and len(ocr_result) >= 1:
            result = ocr_result[0]
        else:
            result = ocr_result
        
        # Parse OCR results
        extracted_texts = []
        
        if result and len(result) > 0:
            for line in result:
                if line and len(line) >= 2:
                    text = line[1] if isinstance(line[1], str) else str(line[1])
                    if text:
                        extracted_texts.append(text)
        
        # Join all extracted text lines
        full_text = "\n".join(extracted_texts) if extracted_texts else ""
        
        if not full_text:
            return ToolResponse(
                content=[TextBlock(
                    type="text",
                    text="Warning: No text was extracted from the image."
                )]
            )
        
        return ToolResponse(
            content=[TextBlock(
                type="text",
                text=full_text
            )]
        )
        
    except Exception as e:
        error_msg = str(e)
        help_text = f"Error during OCR processing: {error_msg}"
        
        return ToolResponse(
            content=[TextBlock(
                type="text",
                text=help_text
            )]
        )


def detect_test_blocks_with_ocr(img: np.ndarray, image_path: str) -> List[Dict[str, Any]]:
    """
    使用 OCR 检测图像中的测试块。
    
    医学报告通常包含多个测试结果，每个测试块包含：
    - 测试名称和时间
    - 操作者信息
    - 测试数据/图表
    
    Args:
        img: 输入图像 (BGR 格式)
        image_path: 图像路径
        
    Returns:
        测试块列表，每个包含 bbox、title、ocr_text 等信息
    """
    h, w = img.shape[:2]
    test_blocks = []
    
    # 获取 OCR 结果
    ocr = get_ocr_instance()
    if ocr is None:
        return test_blocks
    
    ocr_result = ocr(image_path)
    if not ocr_result or not ocr_result[0]:
        return test_blocks
    
    # 收集所有文本行及其位置
    text_lines = []
    for line in ocr_result[0]:
        box = line[0]
        text = line[1] if isinstance(line[1], str) else str(line[1])
        confidence = line[2] if len(line) > 2 else 1.0
        
        # 计算 Y 坐标
        y_top = min(p[1] for p in box)
        y_bottom = max(p[1] for p in box)
        x_left = min(p[0] for p in box)
        x_right = max(p[0] for p in box)
        
        text_lines.append({
            'text': text,
            'y_top': y_top,
            'y_bottom': y_bottom,
            'x_left': x_left,
            'x_right': x_right,
            'y_center': (y_top + y_bottom) / 2,
            'confidence': confidence
        })
    
    # 按 Y 坐标排序
    text_lines.sort(key=lambda x: x['y_top'])
    
    # 识别测试块的起始位置（通过关键词）
    test_start_patterns = [
        r'Dix-Hallpike.*测试',  # Dix-Hallpike 测试
        r'Roll.*测试',          # Roll 测试
        r'.*测试：\d{4}',       # 带日期的测试
        r'.*Test.*\d{4}',       # 英文测试
        r'^位置性',             # 位置性标题
    ]
    
    test_start_indices = []
    for i, line in enumerate(text_lines):
        text = line['text']
        for pattern in test_start_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # 检查是否是新测试的开始（不是图表标签）
                if '遮光' not in text and '水平' not in text[-5:]:
                    test_start_indices.append(i)
                    break
    
    # 如果没有检测到测试块，尝试使用间距分析
    if len(test_start_indices) < 2:
        # 分析文本行之间的间距
        large_gaps = []
        for i in range(1, len(text_lines)):
            gap = text_lines[i]['y_top'] - text_lines[i-1]['y_bottom']
            if gap > 50:  # 大间距可能是测试块之间的分隔
                large_gaps.append((i, gap))
        
        if large_gaps:
            test_start_indices = [0] + [g[0] for g in large_gaps]
    
    # 根据起始位置划分测试块
    if not test_start_indices:
        test_start_indices = [0]
    
    # 创建测试块
    for i, start_idx in enumerate(test_start_indices):
        # 确定结束位置
        if i + 1 < len(test_start_indices):
            end_idx = test_start_indices[i + 1]
        else:
            end_idx = len(text_lines)
        
        if end_idx <= start_idx:
            continue
        
        # 获取该块的所有文本
        block_lines = text_lines[start_idx:end_idx]
        if not block_lines:
            continue
        
        # 计算边界
        y_start = max(0, int(block_lines[0]['y_top']) - 10)
        y_end = min(h, int(block_lines[-1]['y_bottom']) + 10)
        
        # 如果是第一个块，可能需要包含顶部标题
        if i == 0 and start_idx == 0:
            y_start = 0
        
        # 如果是最后一个块，扩展到图像底部
        if i == len(test_start_indices) - 1:
            y_end = h
        
        # 合并该块的文本
        block_text = '\n'.join([line['text'] for line in block_lines])
        
        # 提取标题
        title = block_lines[0]['text'] if block_lines else "未知测试"
        
        test_blocks.append({
            'index': i + 1,
            'title': title,
            'bbox': [0, y_start, w, y_end],
            'y_start': y_start,
            'y_end': y_end,
            'width': w,
            'height': y_end - y_start,
            'ocr_text': block_text,
            'text_lines_count': len(block_lines)
        })
    
    return test_blocks


def detect_chart_regions_smart(img: np.ndarray, image_path: str) -> List[Dict[str, Any]]:
    """
    智能检测图表区域，使用 OCR 辅助定位。
    
    Args:
        img: 输入图像 (BGR 格式)
        image_path: 图像路径
        
    Returns:
        检测到的图表区域列表
    """
    h, w = img.shape[:2]
    
    # 首先尝试使用 OCR 检测测试块
    test_blocks = detect_test_blocks_with_ocr(img, image_path)
    
    if test_blocks and len(test_blocks) > 1:
        # 使用 OCR 检测到的测试块
        return test_blocks
    
    # 回退到基于图像分析的方法
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 使用水平投影分析
    # 计算每行的平均灰度
    row_means = np.mean(gray, axis=1)
    
    # 使用滑动窗口检测亮度变化
    window_size = 20
    row_diffs = []
    for i in range(window_size, h - window_size):
        top_mean = np.mean(row_means[i-window_size:i])
        bottom_mean = np.mean(row_means[i:i+window_size])
        row_diffs.append(abs(top_mean - bottom_mean))
    
    row_diffs = np.array(row_diffs)
    
    # 找到显著的分割点
    threshold = np.mean(row_diffs) + 2 * np.std(row_diffs)
    potential_splits = []
    
    for i, diff in enumerate(row_diffs):
        if diff > threshold:
            actual_y = i + window_size
            # 避免太靠近的分割点
            if not potential_splits or actual_y - potential_splits[-1] > 100:
                potential_splits.append(actual_y)
    
    # 如果找到分割点，创建区域
    charts = []
    if potential_splits:
        # 添加起始和结束
        all_splits = [0] + potential_splits + [h]
        
        for i in range(len(all_splits) - 1):
            y_start = all_splits[i]
            y_end = all_splits[i + 1]
            
            if y_end - y_start > 50:  # 最小高度
                charts.append({
                    'index': i + 1,
                    'title': f'区域 {i + 1}',
                    'bbox': [0, y_start, w, y_end],
                    'y_start': y_start,
                    'y_end': y_end,
                    'width': w,
                    'height': y_end - y_start,
                    'ocr_text': '',
                    'text_lines_count': 0
                })
    
    # 如果没有检测到分割点，将整个图像作为一个区域
    if not charts:
        charts.append({
            'index': 1,
            'title': '完整图像',
            'bbox': [0, 0, w, h],
            'y_start': 0,
            'y_end': h,
            'width': w,
            'height': h,
            'ocr_text': '',
            'text_lines_count': 0
        })
    
    return charts


def extract_with_layout_analysis(
    image_path: str,
    output_dir: Optional[str] = None,
    save_figures: bool = True
) -> Dict[str, Any]:
    """
    提取图像中的文本和图表区域，使用智能版面分析。
    
    对于包含多个测试结果的医学报告图像，会：
    1. 识别每个测试块的边界
    2. 提取每个测试块的 OCR 文本
    3. 保存每个测试块为独立图像
    4. 建立原图和分割图的关联
    
    Args:
        image_path: 图像文件路径
        output_dir: 保存分割图像的目录
        save_figures: 是否保存分割的图像
        
    Returns:
        包含以下字段的字典:
        - ocr_text: 完整 OCR 文本
        - layout_regions: 版面区域列表
        - extracted_figures: 提取的图像信息列表（包含原图关联）
        - source_image: 原图信息
    """
    result = {
        "ocr_text": "",
        "layout_regions": [],
        "extracted_figures": [],
        "source_image": None,
        "error": None
    }
    
    if not os.path.exists(image_path):
        result["error"] = f"File not found: {image_path}"
        return result
    
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            result["error"] = f"Failed to read image: {image_path}"
            return result
        
        img_height, img_width = img.shape[:2]
        
        # 保存原图信息
        result["source_image"] = {
            "file_name": os.path.basename(image_path),
            "file_path": image_path,
            "width": img_width,
            "height": img_height
        }
        
        # 首先获取完整的 OCR 文本
        ocr_result = extract_text_from_image(image_path)
        if ocr_result.content:
            result["ocr_text"] = ocr_result.content[0].get("text", "")
        
        # 对于 figure 图像进行智能分割
        if is_figure_path(image_path) and save_figures and output_dir:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 使用智能检测
            charts = detect_chart_regions_smart(img, image_path)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            for chart in charts:
                y_start = chart['y_start']
                y_end = chart['y_end']
                
                # 提取图表区域
                chart_img = img[y_start:y_end, :]
                
                # 生成文件名
                chart_filename = f"{base_name}_block_{chart['index']}.jpg"
                chart_path = os.path.join(output_dir, chart_filename)
                
                # 保存图表
                cv2.imwrite(chart_path, chart_img)
                
                # 如果该块没有 OCR 文本，单独获取
                if not chart.get('ocr_text'):
                    block_ocr = extract_text_from_image(chart_path)
                    if block_ocr.content:
                        chart['ocr_text'] = block_ocr.content[0].get("text", "")
                
                # 构建图像信息（包含原图关联）
                figure_info = {
                    "index": chart['index'],
                    "type": "test_block",
                    "title": chart.get('title', ''),
                    "source_file": os.path.basename(image_path),
                    "source_image": {
                        "file_name": os.path.basename(image_path),
                        "file_path": image_path,
                        "width": img_width,
                        "height": img_height
                    },
                    "extracted_file": chart_filename,
                    "extracted_path": chart_path,
                    "bbox": chart['bbox'],
                    "y_start": y_start,
                    "y_end": y_end,
                    "width": chart['width'],
                    "height": chart['height'],
                    "ocr_text": chart.get('ocr_text', ''),
                    "text_lines_count": chart.get('text_lines_count', 0)
                }
                result["extracted_figures"].append(figure_info)
                
                # 添加到版面区域
                result["layout_regions"].append({
                    "type": "test_block",
                    "index": chart['index'],
                    "title": chart.get('title', ''),
                    "bbox": chart['bbox'],
                    "content": figure_info
                })
            
            # 添加原图完整文本区域
            if result["ocr_text"]:
                result["layout_regions"].append({
                    "type": "full_text",
                    "bbox": [0, 0, img_width, img_height],
                    "content": result["ocr_text"]
                })
        
    except Exception as e:
        result["error"] = str(e)
        import traceback
        result["error_traceback"] = traceback.format_exc()
        # 回退到简单 OCR
        try:
            ocr_result = extract_text_from_image(image_path)
            if ocr_result.content:
                result["ocr_text"] = ocr_result.content[0].get("text", "")
        except:
            pass
    
    return result


def extract_ocr_with_layout(
    image_path: str,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    智能 OCR 提取，对 figure 图像使用版面分析。
    
    Args:
        image_path: 图像文件路径
        output_dir: 保存分割图像的目录（对于 figure 图像）
        
    Returns:
        包含 OCR 结果和分割图像信息的字典
    """
    # 检查是否需要版面分析
    needs_layout_analysis = is_figure_path(image_path)
    
    if needs_layout_analysis:
        # 使用版面分析
        return extract_with_layout_analysis(
            image_path,
            output_dir=output_dir,
            save_figures=True
        )
    else:
        # 简单 OCR
        result = {
            "ocr_text": "",
            "layout_regions": [],
            "extracted_figures": [],
            "source_image": None,
            "error": None
        }
        
        ocr_result = extract_text_from_image(image_path)
        if ocr_result.content:
            result["ocr_text"] = ocr_result.content[0].get("text", "")
        
        return result


def is_image_file(file_path: str) -> bool:
    """
    Check if a file path points to an image file.
    
    Args:
        file_path (str): Path to check.
    
    Returns:
        bool: True if the file is an image, False otherwise.
    """
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return False
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
    _, ext = os.path.splitext(file_path.lower())
    return ext in image_extensions
