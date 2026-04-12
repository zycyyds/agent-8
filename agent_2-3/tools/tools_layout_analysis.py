"""
版面分析工具 - 用于分析医学图像中的布局并切割子区域

支持的切割方法：
1. YOLO 目标检测（如果模型可用）
2. 基于分割线检测（水平/垂直明显分割线）
3. 基于投影分析（像素投影找分割点）
4. 基于连通组件分析
5. 智能网格切割（根据内容密度）
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


# 尝试导入 YOLO
YOLO_AVAILABLE = False
yolo_model = None

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass


def is_figure_path(file_path: str) -> bool:
    """
    检测文件路径是否来自 figure 文件夹
    
    Args:
        file_path: 文件路径
        
    Returns:
        是否来自 figure 文件夹
    """
    path_parts = os.path.normpath(file_path).split(os.sep)
    
    for part in path_parts:
        if part.lower() == 'figure':
            return True
    
    return False


def load_yolo_model(model_path: str = None) -> bool:
    """
    加载 YOLO 模型
    
    Args:
        model_path: 模型路径，默认使用 yolov8n
        
    Returns:
        是否加载成功
    """
    global yolo_model, YOLO_AVAILABLE
    
    if not YOLO_AVAILABLE:
        return False
    
    try:
        if model_path and os.path.exists(model_path):
            yolo_model = YOLO(model_path)
        else:
            # 使用预训练模型
            yolo_model = YOLO('yolov8n.pt')
        return True
    except Exception as e:
        print(f"YOLO模型加载失败: {e}")
        return False


def analyze_layout(
    image_path: str,
    output_dir: Optional[str] = None,
    min_region_area: int = 3000,
    padding: int = 5,
    save_regions: bool = True,
    method: str = "auto"
) -> Dict[str, Any]:
    """
    分析图像布局，检测和切割子区域
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        min_region_area: 最小区域面积阈值
        padding: 切割时的边距
        save_regions: 是否保存切割的区域
        method: 切割方法 ('auto', 'yolo', 'line', 'projection', 'contour', 'grid')
        
    Returns:
        包含区域信息的字典
    """
    result = {
        "success": False,
        "image_path": image_path,
        "regions": [],
        "saved_files": [],
        "error": None
    }
    
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            result["error"] = f"无法读取图像: {image_path}"
            return result
        
        height, width = image.shape[:2]
        result["image_size"] = {"width": width, "height": height}
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 根据方法选择切割策略
        if method == "auto":
            # 自动选择最佳方法
            regions, detection_method = _auto_segment(image, gray, height, width, min_region_area)
        elif method == "yolo":
            regions = _yolo_segment(image_path, image, height, width)
            detection_method = "yolo"
        elif method == "line":
            regions = _line_based_segment(image, gray, height, width)
            detection_method = "line"
        elif method == "projection":
            regions = _projection_segment(image, gray, height, width)
            detection_method = "projection"
        elif method == "contour":
            regions = _contour_segment(image, gray, height, width, min_region_area)
            detection_method = "contour"
        else:
            regions = _smart_grid_segment(image, gray, height, width)
            detection_method = "smart_grid"
        
        result["regions"] = regions
        result["detection_method"] = detection_method
        
        # 保存切割的区域
        if save_regions and len(regions) > 0:
            if output_dir is None:
                base_dir = os.path.dirname(image_path)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_dir = os.path.join(base_dir, f"{base_name}_segments")
            
            os.makedirs(output_dir, exist_ok=True)
            
            for region in regions:
                x, y, w, h = region["bbox"]
                
                # 添加 padding
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(width, x + w + padding)
                y2 = min(height, y + h + padding)
                
                # 切割区域
                segment = image[y1:y2, x1:x2]
                
                # 保存
                region_type = region.get("type", "region")
                segment_filename = f"segment_{region['index']:02d}_{region_type}.jpg"
                segment_path = os.path.join(output_dir, segment_filename)
                cv2.imwrite(segment_path, segment)
                
                region["saved_path"] = segment_path
                result["saved_files"].append(segment_path)
        
        result["success"] = True
        result["region_count"] = len(regions)
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def _auto_segment(
    image: np.ndarray, 
    gray: np.ndarray, 
    height: int, 
    width: int,
    min_area: int
) -> Tuple[List[Dict], str]:
    """
    自动选择最佳切割方法
    
    优先级：
    1. YOLO（如果可用且检测到对象）
    2. 医学图像专用方法（检测独立的内容块）
    3. 分割线检测
    4. 轮廓检测
    5. 智能网格
    """
    # 1. 尝试 YOLO
    if YOLO_AVAILABLE and yolo_model is not None:
        yolo_regions = _yolo_segment(None, image, height, width)
        if len(yolo_regions) >= 2:
            return yolo_regions, "yolo"
    
    # 2. 尝试医学图像专用方法
    medical_regions = _medical_image_segment(image, gray, height, width, min_area)
    if len(medical_regions) >= 2:
        # 验证切割质量：每个区域的面积应该足够大
        avg_area = sum(r["area"] for r in medical_regions) / len(medical_regions)
        min_acceptable_area = height * width * 0.08  # 至少占总面积的8%
        if avg_area >= min_acceptable_area:
            return medical_regions, "medical"
    
    # 3. 尝试基于分割线的方法
    line_regions = _line_based_segment(image, gray, height, width)
    if len(line_regions) >= 2:
        # 同样验证切割质量
        avg_height = sum(r["bbox"][3] for r in line_regions) / len(line_regions)
        if avg_height >= height * 0.15:  # 平均高度至少是原图的15%
            return line_regions, "line"
    
    # 4. 尝试轮廓检测
    contour_regions = _contour_segment(image, gray, height, width, min_area)
    if len(contour_regions) >= 2:
        return contour_regions, "contour"
    
    # 5. 使用智能网格（默认2x2或根据内容决定）
    grid_regions = _smart_grid_segment(image, gray, height, width)
    return grid_regions, "smart_grid"


def _medical_image_segment(
    image: np.ndarray,
    gray: np.ndarray,
    height: int,
    width: int,
    min_area: int
) -> List[Dict[str, Any]]:
    """
    专门针对医学图像的切割方法
    适用于包含多个独立图表/波形的检查报告
    
    策略：
    1. 检测明显的矩形边框区域
    2. 如果有多个独立的内容块，按块切割
    3. 否则检测主要的水平分隔线，按区域切割
    """
    regions = []
    
    # 方法1：检测矩形边框
    rect_regions = _detect_rectangle_regions(image, gray, height, width, min_area)
    if len(rect_regions) >= 2:
        return rect_regions
    
    # 方法2：基于颜色/亮度差异检测独立区域
    color_regions = _detect_color_regions(image, gray, height, width, min_area)
    if len(color_regions) >= 2:
        return color_regions
    
    # 方法3：检测主要的水平分隔区域（大间隙）
    gap_regions = _detect_gap_regions(image, gray, height, width)
    if len(gap_regions) >= 2:
        return gap_regions
    
    return regions


def _detect_rectangle_regions(
    image: np.ndarray,
    gray: np.ndarray,
    height: int,
    width: int,
    min_area: int
) -> List[Dict[str, Any]]:
    """检测图像中的矩形边框区域"""
    regions = []
    
    # 边缘检测
    edges = cv2.Canny(gray, 30, 100)
    
    # 膨胀边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # 近似多边形
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 检查是否接近矩形（4个顶点）
        if len(approx) >= 4 and len(approx) <= 6:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # 面积和比例检查
            if area < min_area:
                continue
            if w < width * 0.2 or h < height * 0.1:
                continue
            if w > width * 0.95 and h > height * 0.95:
                continue
            
            # 检查填充率（矩形内的内容比例）
            roi = gray[y:y+h, x:x+w]
            non_white = np.sum(roi < 240) / roi.size
            if non_white < 0.02:  # 太空白，跳过
                continue
            
            regions.append({
                "bbox": (x, y, w, h),
                "area": area,
                "center": (x + w // 2, y + h // 2)
            })
    
    # 合并重叠区域
    regions = _merge_overlapping_regions(regions, overlap_threshold=0.5)
    
    # 排序并添加索引
    regions.sort(key=lambda r: (r["center"][1], r["center"][0]))
    for idx, r in enumerate(regions):
        r["index"] = idx
        r["type"] = _classify_region(image, r["bbox"])
    
    return regions


def _detect_color_regions(
    image: np.ndarray,
    gray: np.ndarray,
    height: int,
    width: int,
    min_area: int
) -> List[Dict[str, Any]]:
    """基于颜色/亮度差异检测独立的内容区域"""
    regions = []
    
    # 使用多种阈值方法
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 21, 5)
    
    # 合并结果
    binary = cv2.bitwise_or(binary_otsu, binary_adapt)
    
    # 形态学操作：连接相邻的内容，但保持区域分离
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    dilated = cv2.dilate(binary, kernel_dilate, iterations=2)
    
    # 腐蚀回来一点，避免区域太大
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.erode(dilated, kernel_erode, iterations=1)
    
    # 查找连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    
    for i in range(1, num_labels):  # 跳过背景
        x, y, w, h, area = stats[i]
        
        if area < min_area:
            continue
        if w < width * 0.15 or h < height * 0.1:
            continue
        if w > width * 0.95 and h > height * 0.95:
            continue
        
        regions.append({
            "bbox": (x, y, w, h),
            "area": area,
            "center": (int(centroids[i][0]), int(centroids[i][1]))
        })
    
    # 合并重叠区域
    regions = _merge_overlapping_regions(regions, overlap_threshold=0.3)
    
    # 排序
    regions.sort(key=lambda r: (r["center"][1], r["center"][0]))
    for idx, r in enumerate(regions):
        r["index"] = idx
        r["type"] = _classify_region(image, r["bbox"])
    
    return regions


def _detect_gap_regions(
    image: np.ndarray,
    gray: np.ndarray,
    height: int,
    width: int,
    min_gap_height: int = None
) -> List[Dict[str, Any]]:
    """
    检测水平方向的大间隙，按间隙切割
    适用于有明显水平分隔的报告
    """
    regions = []
    
    if min_gap_height is None:
        min_gap_height = height * 0.03  # 间隙至少是图像高度的3%
    
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 计算每行的内容密度
    row_density = np.sum(binary, axis=1) / width
    
    # 平滑
    kernel_size = max(3, height // 100)
    row_density_smooth = np.convolve(row_density, np.ones(kernel_size)/kernel_size, mode='same')
    
    # 找到低密度区域（间隙）
    threshold = np.mean(row_density_smooth) * 0.1
    
    gaps = []
    in_gap = False
    gap_start = 0
    
    for i in range(len(row_density_smooth)):
        if row_density_smooth[i] < threshold:
            if not in_gap:
                in_gap = True
                gap_start = i
        else:
            if in_gap:
                gap_end = i
                gap_height = gap_end - gap_start
                if gap_height >= min_gap_height:
                    gaps.append((gap_start, gap_end))
                in_gap = False
    
    # 根据间隙创建区域
    if len(gaps) >= 1:
        # 添加边界
        boundaries = [0]
        for gap_start, gap_end in gaps:
            boundaries.append((gap_start + gap_end) // 2)
        boundaries.append(height)
        
        idx = 0
        for i in range(len(boundaries) - 1):
            y1, y2 = boundaries[i], boundaries[i+1]
            h = y2 - y1
            
            # 跳过太小的区域
            if h < height * 0.1:
                continue
            
            # 检查区域是否有内容
            roi = image[y1:y2, 0:width]
            if _has_content(roi):
                regions.append({
                    "index": idx,
                    "bbox": (0, y1, width, h),
                    "area": width * h,
                    "type": _classify_region(image, (0, y1, width, h))
                })
                idx += 1
    
    return regions


def _yolo_segment(
    image_path: Optional[str],
    image: np.ndarray,
    height: int,
    width: int,
    conf_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """
    使用 YOLO 进行目标检测和切割
    """
    global yolo_model
    regions = []
    
    if not YOLO_AVAILABLE:
        return regions
    
    # 如果模型未加载，尝试加载
    if yolo_model is None:
        if not load_yolo_model():
            return regions
    
    try:
        # 运行推理
        results = yolo_model(image, conf=conf_threshold, verbose=False)
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                w, h = x2 - x1, y2 - y1
                
                # 过滤太小的区域
                if w < 50 or h < 50:
                    continue
                
                regions.append({
                    "index": idx,
                    "bbox": (x1, y1, w, h),
                    "area": w * h,
                    "type": f"yolo_cls{cls}",
                    "confidence": conf
                })
        
        # 按位置排序
        regions.sort(key=lambda r: (r["bbox"][1] // 100, r["bbox"][0]))
        for idx, r in enumerate(regions):
            r["index"] = idx
            
    except Exception as e:
        print(f"YOLO检测失败: {e}")
    
    return regions


def _line_based_segment(
    image: np.ndarray,
    gray: np.ndarray,
    height: int,
    width: int,
    line_threshold: int = 100
) -> List[Dict[str, Any]]:
    """
    基于分割线检测的切割方法
    检测图像中明显的水平和垂直分割线
    """
    regions = []
    
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)
    
    # 膨胀边缘以连接断开的线段
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 检测直线 - 使用更宽松的参数
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50,
        minLineLength=min(width, height) * 0.2,
        maxLineGap=30
    )
    
    if lines is None:
        return regions
    
    # 分离水平线和垂直线
    h_lines = []  # 水平分割线的 y 坐标
    v_lines = []  # 垂直分割线的 x 坐标
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # 计算线的长度和角度
        line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        if abs(x2 - x1) < 1:
            angle = 90
        else:
            angle = abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
        
        # 水平线（角度接近0度）
        if angle < 15 and line_length > width * 0.3:
            avg_y = (y1 + y2) // 2
            # 排除边缘线
            if avg_y > height * 0.08 and avg_y < height * 0.92:
                h_lines.append(avg_y)
        
        # 垂直线（角度接近90度）
        if angle > 75 and line_length > height * 0.3:
            avg_x = (x1 + x2) // 2
            # 排除边缘线
            if avg_x > width * 0.08 and avg_x < width * 0.92:
                v_lines.append(avg_x)
    
    # 聚类合并相近的线
    h_lines = _cluster_lines(h_lines, threshold=height * 0.05)
    v_lines = _cluster_lines(v_lines, threshold=width * 0.05)
    
    # 根据分割线创建区域
    h_splits = [0] + sorted(h_lines) + [height]
    v_splits = [0] + sorted(v_lines) + [width]
    
    idx = 0
    for i in range(len(h_splits) - 1):
        for j in range(len(v_splits) - 1):
            y1, y2 = h_splits[i], h_splits[i+1]
            x1, x2 = v_splits[j], v_splits[j+1]
            
            w, h = x2 - x1, y2 - y1
            
            # 跳过太小的区域
            if w < width * 0.12 or h < height * 0.12:
                continue
            
            # 检查区域是否有内容
            cell = image[y1:y2, x1:x2]
            if _has_content(cell):
                regions.append({
                    "index": idx,
                    "bbox": (x1, y1, w, h),
                    "area": w * h,
                    "type": _classify_region(image, (x1, y1, w, h))
                })
                idx += 1
    
    return regions


def _cluster_lines(lines: List[int], threshold: int) -> List[int]:
    """聚类合并相近的线"""
    if not lines:
        return []
    
    lines = sorted(lines)
    clustered = []
    current_cluster = [lines[0]]
    
    for i in range(1, len(lines)):
        if lines[i] - current_cluster[-1] < threshold:
            current_cluster.append(lines[i])
        else:
            # 取聚类的中间值
            clustered.append(int(np.mean(current_cluster)))
            current_cluster = [lines[i]]
    
    clustered.append(int(np.mean(current_cluster)))
    return clustered


def _projection_segment(
    image: np.ndarray,
    gray: np.ndarray,
    height: int,
    width: int
) -> List[Dict[str, Any]]:
    """
    基于投影分析的切割方法
    分析水平和垂直方向的像素投影，找到分割点
    """
    regions = []
    
    # 二值化 - 使用自适应阈值
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 5
    )
    
    # 水平投影（每行的非零像素数）
    h_proj = np.sum(binary, axis=1)
    
    # 垂直投影（每列的非零像素数）
    v_proj = np.sum(binary, axis=0)
    
    # 平滑投影曲线
    kernel_size = max(5, min(height, width) // 50)
    h_proj_smooth = np.convolve(h_proj, np.ones(kernel_size)/kernel_size, mode='same')
    v_proj_smooth = np.convolve(v_proj, np.ones(kernel_size)/kernel_size, mode='same')
    
    # 找到水平分割点（投影值低的位置）
    h_splits = _find_projection_splits(h_proj_smooth, height, min_gap=height * 0.1)
    
    # 找到垂直分割点
    v_splits = _find_projection_splits(v_proj_smooth, width, min_gap=width * 0.1)
    
    # 创建区域
    h_bounds = [0] + h_splits + [height]
    v_bounds = [0] + v_splits + [width]
    
    idx = 0
    for i in range(len(h_bounds) - 1):
        for j in range(len(v_bounds) - 1):
            y1, y2 = h_bounds[i], h_bounds[i+1]
            x1, x2 = v_bounds[j], v_bounds[j+1]
            
            w, h = x2 - x1, y2 - y1
            
            # 跳过太小的区域
            if w < width * 0.12 or h < height * 0.12:
                continue
            
            cell = image[y1:y2, x1:x2]
            if _has_content(cell):
                regions.append({
                    "index": idx,
                    "bbox": (x1, y1, w, h),
                    "area": w * h,
                    "type": _classify_region(image, (x1, y1, w, h))
                })
                idx += 1
    
    return regions


def _find_projection_splits(
    projection: np.ndarray, 
    total_size: int,
    min_gap: int = 20
) -> List[int]:
    """找到投影中的分割点（低谷位置）"""
    splits = []
    
    # 计算投影的平均值和阈值
    mean_proj = np.mean(projection)
    std_proj = np.std(projection)
    
    # 阈值：低于平均值减去一定标准差的位置可能是分割点
    threshold = max(mean_proj * 0.15, mean_proj - std_proj * 0.5)
    
    # 找到连续的低值区域
    in_gap = False
    gap_start = 0
    
    for i in range(len(projection)):
        if projection[i] < threshold:
            if not in_gap:
                in_gap = True
                gap_start = i
        else:
            if in_gap:
                gap_end = i
                gap_width = gap_end - gap_start
                
                # 间隙要足够宽
                if gap_width > total_size * 0.02:
                    gap_center = (gap_start + gap_end) // 2
                    
                    # 确保分割点不在边缘，且与上一个分割点距离足够
                    if gap_center > total_size * 0.08 and gap_center < total_size * 0.92:
                        if not splits or (gap_center - splits[-1]) > min_gap:
                            splits.append(gap_center)
                
                in_gap = False
    
    return splits


def _contour_segment(
    image: np.ndarray,
    gray: np.ndarray,
    height: int,
    width: int,
    min_area: int
) -> List[Dict[str, Any]]:
    """
    基于轮廓检测的切割方法
    使用形态学操作和轮廓检测找到独立的内容区域
    """
    regions = []
    
    # 多种二值化方法尝试
    # 方法1：自适应阈值
    binary1 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 方法2：Otsu阈值
    _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 合并两种结果
    binary = cv2.bitwise_or(binary1, binary2)
    
    # 形态学操作：先闭运算填充小孔，再膨胀连接相邻区域
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    dilated = cv2.dilate(binary, kernel_dilate, iterations=2)
    
    # 查找轮廓
    contours, hierarchy = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 过滤和处理轮廓
    valid_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # 过滤太小或太大的区域
        if w < 50 or h < 50:
            continue
        if w > width * 0.95 and h > height * 0.95:
            continue
        
        # 计算填充率
        rect_area = w * h
        fill_ratio = area / rect_area if rect_area > 0 else 0
        
        valid_regions.append({
            "bbox": (x, y, w, h),
            "area": area,
            "center": (x + w // 2, y + h // 2),
            "fill_ratio": fill_ratio
        })
    
    # 合并重叠的区域
    valid_regions = _merge_overlapping_regions(valid_regions, overlap_threshold=0.3)
    
    # 按位置排序（从上到下，从左到右）
    valid_regions.sort(key=lambda r: (r["center"][1] // 100, r["center"][0]))
    
    for idx, region in enumerate(valid_regions):
        regions.append({
            "index": idx,
            "bbox": region["bbox"],
            "area": region["area"],
            "type": _classify_region(image, region["bbox"])
        })
    
    return regions


def _merge_overlapping_regions(
    regions: List[Dict], 
    overlap_threshold: float = 0.3
) -> List[Dict]:
    """合并重叠的区域"""
    if len(regions) <= 1:
        return regions
    
    merged = []
    used = [False] * len(regions)
    
    for i in range(len(regions)):
        if used[i]:
            continue
        
        current = regions[i].copy()
        x1, y1, w1, h1 = current["bbox"]
        
        for j in range(i + 1, len(regions)):
            if used[j]:
                continue
            
            x2, y2, w2, h2 = regions[j]["bbox"]
            
            # 计算重叠
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y
            
            min_area = min(w1 * h1, w2 * h2)
            
            if min_area > 0 and overlap_area / min_area > overlap_threshold:
                # 合并区域
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                
                current["bbox"] = (new_x, new_y, new_w, new_h)
                current["area"] = new_w * new_h
                current["center"] = (new_x + new_w // 2, new_y + new_h // 2)
                
                x1, y1, w1, h1 = new_x, new_y, new_w, new_h
                used[j] = True
        
        merged.append(current)
    
    return merged


def _smart_grid_segment(
    image: np.ndarray,
    gray: np.ndarray,
    height: int,
    width: int
) -> List[Dict[str, Any]]:
    """
    智能网格切割
    根据图像内容分布自动确定最佳网格配置
    """
    regions = []
    
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 计算不同网格配置下的内容分布
    best_config = (2, 2)  # 默认 2x2
    best_score = 0
    
    for rows in [1, 2, 3, 4]:
        for cols in [1, 2, 3, 4]:
            if rows * cols < 2 or rows * cols > 9:
                continue
            
            score = _evaluate_grid_config(binary, height, width, rows, cols)
            if score > best_score:
                best_score = score
                best_config = (rows, cols)
    
    rows, cols = best_config
    
    # 使用最佳网格配置切割
    cell_h = height // rows
    cell_w = width // cols
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            x = c * cell_w
            y = r * cell_h
            w = cell_w if c < cols - 1 else width - x
            h = cell_h if r < rows - 1 else height - y
            
            cell = image[y:y+h, x:x+w]
            if _has_content(cell, threshold=0.01):
                regions.append({
                    "index": idx,
                    "bbox": (x, y, w, h),
                    "area": w * h,
                    "type": _classify_region(image, (x, y, w, h))
                })
                idx += 1
    
    return regions


def _evaluate_grid_config(
    binary: np.ndarray, 
    height: int, 
    width: int, 
    rows: int, 
    cols: int
) -> float:
    """评估网格配置的质量"""
    cell_h = height // rows
    cell_w = width // cols
    
    # 计算每个单元格的内容密度
    densities = []
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            
            cell = binary[y1:y2, x1:x2]
            density = np.sum(cell > 0) / cell.size if cell.size > 0 else 0
            densities.append(density)
    
    # 好的配置应该有：
    # 1. 每个单元格都有内容（密度不为0）
    # 2. 单元格之间的密度差异不太大
    
    non_empty = sum(1 for d in densities if d > 0.01)
    if non_empty < len(densities) * 0.5:
        return 0
    
    # 计算密度方差（越小越好）
    density_var = np.var(densities) if densities else 1
    
    # 分数 = 非空单元格比例 / (1 + 密度方差)
    score = non_empty / len(densities) / (1 + density_var * 10)
    
    # 偏好较少的切割（避免过度切割）
    score *= (1 - 0.05 * (rows * cols - 2))
    
    return score


def _has_content(image: np.ndarray, threshold: float = 0.02) -> bool:
    """检查图像区域是否有内容（非空白）"""
    if image is None or image.size == 0:
        return False
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # 计算非白色像素比例
    non_white = np.sum(gray < 240)
    total = gray.size
    
    return (non_white / total) > threshold


def _classify_region(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
    """根据区域特征分类区域类型"""
    x, y, w, h = bbox
    region = image[y:y+h, x:x+w]
    
    if region is None or region.size == 0:
        return "unknown"
    
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # 直线检测
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    line_count = len(lines) if lines is not None else 0
    
    # 计算颜色特征
    if len(region.shape) == 3:
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        color_std = np.std(hsv[:, :, 0])
        saturation_mean = np.mean(hsv[:, :, 1])
    else:
        color_std = 0
        saturation_mean = 0
    
    # 分类规则
    # 表格：大量直线
    if line_count > 15:
        return "table"
    
    # 图表/波形：中等边缘密度，有颜色变化
    if edge_density > 0.03 and (color_std > 15 or saturation_mean > 30):
        return "chart"
    
    # 文本区域：低边缘密度，低饱和度
    if edge_density < 0.02 and saturation_mean < 20:
        return "text"
    
    # 图像：高颜色复杂度
    if color_std > 25 or saturation_mean > 50:
        return "figure"
    
    return "content"


def analyze_medical_figure(
    image_path: str,
    output_dir: Optional[str] = None,
    patient_id: Optional[str] = None,
    category: Optional[str] = None,
    method: str = "auto"
) -> Dict[str, Any]:
    """
    分析医学图像（专门针对 figure 文件夹中的图像）
    
    Args:
        image_path: 图像路径
        output_dir: 输出目录
        patient_id: 病人ID
        category: 图像分类
        method: 切割方法
        
    Returns:
        分析结果
    """
    result = {
        "success": False,
        "image_path": image_path,
        "is_figure": is_figure_path(image_path),
        "patient_id": patient_id,
        "category": category,
        "layout_analysis": None,
        "segments": []
    }
    
    if not result["is_figure"]:
        result["error"] = "图像不是来自 figure 文件夹"
        return result
    
    # 设置输出目录
    if output_dir is None:
        base_dir = os.path.dirname(image_path)
        if patient_id:
            output_dir = os.path.join(base_dir, f"{patient_id}_segments")
        else:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = os.path.join(base_dir, f"{base_name}_segments")
    
    # 进行版面分析
    layout_result = analyze_layout(
        image_path=image_path,
        output_dir=output_dir,
        save_regions=True,
        method=method
    )
    
    result["layout_analysis"] = layout_result
    
    if layout_result["success"]:
        result["success"] = True
        result["segments"] = layout_result["saved_files"]
        result["region_count"] = layout_result["region_count"]
        result["detection_method"] = layout_result.get("detection_method", "unknown")
    else:
        result["error"] = layout_result.get("error", "版面分析失败")
    
    return result


def batch_analyze_figures(
    folder_path: str,
    output_base_dir: Optional[str] = None,
    method: str = "auto"
) -> Dict[str, Any]:
    """
    批量分析文件夹中的 figure 图像
    
    Args:
        folder_path: 文件夹路径
        output_base_dir: 输出基础目录
        method: 切割方法
        
    Returns:
        批量分析结果
    """
    result = {
        "success": False,
        "folder_path": folder_path,
        "total_images": 0,
        "analyzed_images": 0,
        "total_segments": 0,
        "results": [],
        "errors": []
    }
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    try:
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in image_extensions:
                    continue
                
                file_path = os.path.join(root, filename)
                result["total_images"] += 1
                
                if not is_figure_path(file_path):
                    continue
                
                if output_base_dir:
                    rel_path = os.path.relpath(root, folder_path)
                    output_dir = os.path.join(output_base_dir, rel_path)
                else:
                    output_dir = None
                
                analysis_result = analyze_medical_figure(
                    image_path=file_path,
                    output_dir=output_dir,
                    method=method
                )
                
                result["results"].append(analysis_result)
                
                if analysis_result["success"]:
                    result["analyzed_images"] += 1
                    result["total_segments"] += len(analysis_result.get("segments", []))
                else:
                    result["errors"].append({
                        "file": file_path,
                        "error": analysis_result.get("error", "未知错误")
                    })
        
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


# 为 AgentScope 工具系统提供接口
def layout_analysis_tool(
    image_path: str, 
    save_segments: bool = True,
    method: str = "auto"
) -> Dict[str, Any]:
    """
    版面分析工具函数（供 Agent 调用）
    
    Args:
        image_path: 图像文件路径
        save_segments: 是否保存切割后的图像
        method: 切割方法 ('auto', 'yolo', 'line', 'projection', 'contour', 'grid')
        
    Returns:
        分析结果字典
    """
    if is_figure_path(image_path):
        return analyze_medical_figure(image_path, method=method)
    else:
        return analyze_layout(image_path, save_regions=save_segments, method=method)
