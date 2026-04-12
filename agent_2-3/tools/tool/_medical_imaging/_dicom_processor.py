# -*- coding: utf-8 -*-
"""医学影像DICOM数据处理工具函数。"""
from pathlib import Path
from typing import Union, Literal, Optional
import numpy as np

try:
    import pydicom
except ImportError:
    pydicom = None

from ...message import TextBlock
from .._response import ToolResponse


def _check_dependencies() -> None:
    """检查必要的依赖是否已安装。"""
    if pydicom is None:
        raise ImportError(
            "pydicom is required for DICOM processing. "
            "Please install it with: pip install pydicom"
        )


def _normalize_to_uint8(
    image: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> np.ndarray:
    """将图像归一化到(0, 255)范围并转换为uint8类型。

    Args:
        image (np.ndarray): 输入图像数组
        min_val (Optional[float]): 最小值，如果为None则使用图像的最小值
        max_val (Optional[float]): 最大值，如果为None则使用图像的最大值

    Returns:
        np.ndarray: 归一化后的uint8图像数组
    """
    image = image.astype(np.float64)

    if min_val is None:
        min_val = np.min(image)
    if max_val is None:
        max_val = np.max(image)

    # 避免除零
    if max_val == min_val:
        return np.zeros_like(image, dtype=np.uint8)

    # 归一化到(0, 1)
    normalized = (image - min_val) / (max_val - min_val)

    # 缩放到(0, 255)并转换为uint8
    normalized = np.clip(normalized * 255, 0, 255)
    return normalized.astype(np.uint8)


def _load_dicom_2d(dicom_path: Union[str, Path]) -> tuple[np.ndarray, dict]:
    """加载单个2D DICOM文件。

    Args:
        dicom_path (Union[str, Path]): DICOM文件路径

    Returns:
        tuple[np.ndarray, dict]: (像素数组, 元数据字典)
    """
    _check_dependencies()
    dicom_path = Path(dicom_path)

    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM文件不存在: {dicom_path}")

    ds = pydicom.dcmread(str(dicom_path))

    # 获取像素数组
    pixel_array = ds.pixel_array.astype(np.float64)

    # 应用窗宽窗位（如果存在）
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        window_center = float(ds.WindowCenter) if hasattr(ds.WindowCenter, "__iter__") else float(ds.WindowCenter[0])
        window_width = float(ds.WindowWidth) if hasattr(ds.WindowWidth, "__iter__") else float(ds.WindowWidth[0])
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2
        pixel_array = np.clip(pixel_array, window_min, window_max)

    # 应用重缩放斜率和截距（如果存在）
    if hasattr(ds, "RescaleSlope"):
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept) if hasattr(ds, "RescaleIntercept") else 0.0
        pixel_array = pixel_array * slope + intercept

    # 提取元数据
    metadata = {
        "PatientID": getattr(ds, "PatientID", "Unknown"),
        "StudyInstanceUID": getattr(ds, "StudyInstanceUID", "Unknown"),
        "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", "Unknown"),
        "InstanceNumber": getattr(ds, "InstanceNumber", None),
        "Modality": getattr(ds, "Modality", "Unknown"),
        "SliceThickness": getattr(ds, "SliceThickness", None),
        "PixelSpacing": getattr(ds, "PixelSpacing", None),
        "ImagePositionPatient": getattr(ds, "ImagePositionPatient", None),
    }

    return pixel_array, metadata


def _load_dicom_3d(dicom_dir: Union[str, Path]) -> tuple[np.ndarray, dict]:
    """加载3D DICOM序列（一个目录中的多个DICOM文件）。

    Args:
        dicom_dir (Union[str, Path]): 包含DICOM文件的目录路径

    Returns:
        tuple[np.ndarray, dict]: (3D像素数组, 元数据字典)
    """
    _check_dependencies()
    dicom_dir = Path(dicom_dir)

    if not dicom_dir.exists() or not dicom_dir.is_dir():
        raise FileNotFoundError(f"DICOM目录不存在: {dicom_dir}")

    # 获取所有DICOM文件
    dicom_files = sorted(dicom_dir.glob("*.dcm"))
    if not dicom_files:
        dicom_files = sorted(dicom_dir.glob("*.dicom"))
    if not dicom_files:
        # 尝试查找所有文件（可能是无扩展名的DICOM文件）
        dicom_files = [f for f in sorted(dicom_dir.iterdir()) if f.is_file()]

    if not dicom_files:
        raise ValueError(f"在目录中未找到DICOM文件: {dicom_dir}")

    # 读取所有DICOM文件
    datasets = []
    for dicom_file in dicom_files:
        try:
            ds = pydicom.dcmread(str(dicom_file))
            datasets.append((dicom_file, ds))
        except Exception as e:
            # 跳过无法读取的文件
            continue

    if not datasets:
        raise ValueError(f"无法读取目录中的任何DICOM文件: {dicom_dir}")

    # 按InstanceNumber或文件名排序
    def get_sort_key(item):
        _, ds = item
        if hasattr(ds, "InstanceNumber") and ds.InstanceNumber is not None:
            return float(ds.InstanceNumber)
        return float("inf")

    datasets.sort(key=get_sort_key)

    # 提取像素数组
    pixel_arrays = []
    metadata_list = []
    for _, ds in datasets:
        pixel_array = ds.pixel_array.astype(np.float64)

        # 应用窗宽窗位
        if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
            window_center = float(ds.WindowCenter) if hasattr(ds.WindowCenter, "__iter__") else float(ds.WindowCenter[0])
            window_width = float(ds.WindowWidth) if hasattr(ds.WindowWidth, "__iter__") else float(ds.WindowWidth[0])
            window_min = window_center - window_width / 2
            window_max = window_center + window_width / 2
            pixel_array = np.clip(pixel_array, window_min, window_max)

        # 应用重缩放
        if hasattr(ds, "RescaleSlope"):
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept) if hasattr(ds, "RescaleIntercept") else 0.0
            pixel_array = pixel_array * slope + intercept

        pixel_arrays.append(pixel_array)
        metadata_list.append({
            "InstanceNumber": getattr(ds, "InstanceNumber", None),
            "ImagePositionPatient": getattr(ds, "ImagePositionPatient", None),
        })

    # 堆叠成3D数组
    volume = np.stack(pixel_arrays, axis=0)

    # 提取第一个文件的通用元数据
    first_ds = datasets[0][1]
    metadata = {
        "PatientID": getattr(first_ds, "PatientID", "Unknown"),
        "StudyInstanceUID": getattr(first_ds, "StudyInstanceUID", "Unknown"),
        "SeriesInstanceUID": getattr(first_ds, "SeriesInstanceUID", "Unknown"),
        "Modality": getattr(first_ds, "Modality", "Unknown"),
        "SliceThickness": getattr(first_ds, "SliceThickness", None),
        "PixelSpacing": getattr(first_ds, "PixelSpacing", None),
        "NumSlices": len(datasets),
        "VolumeShape": volume.shape,
    }

    return volume, metadata


def process_dicom_image(
    dicom_path: Union[str, Path],
    dimension: Literal["2d", "3d", "auto"] = "auto",
    normalize: bool = True,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> ToolResponse:
    """处理DICOM医学影像数据，支持2D和3D处理，并归一化到(0, 255)。

    Args:
        dicom_path (Union[str, Path]): DICOM文件路径（2D）或目录路径（3D）
        dimension (Literal["2d", "3d", "auto"]): 处理维度，"auto"时自动检测
        normalize (bool): 是否归一化到(0, 255)，默认为True
        min_val (Optional[float]): 归一化的最小值，如果为None则使用图像的最小值
        max_val (Optional[float]): 归一化的最大值，如果为None则使用图像的最大值
        output_path (Optional[Union[str, Path]]): 输出路径（.npy格式），如果为None则不保存

    Returns:
        ToolResponse: 包含处理结果的工具响应
    """
    _check_dependencies()
    dicom_path = Path(dicom_path)

    if not dicom_path.exists():
        raise FileNotFoundError(f"路径不存在: {dicom_path}")

    # 自动检测维度
    if dimension == "auto":
        if dicom_path.is_file():
            dimension = "2d"
        elif dicom_path.is_dir():
            dimension = "3d"
        else:
            raise ValueError(f"无法确定路径类型: {dicom_path}")

    # 加载数据
    if dimension == "2d":
        pixel_array, metadata = _load_dicom_2d(dicom_path)
    elif dimension == "3d":
        pixel_array, metadata = _load_dicom_3d(dicom_path)
    else:
        raise ValueError(f"不支持的维度: {dimension}")

    # 归一化
    if normalize:
        pixel_array = _normalize_to_uint8(pixel_array, min_val, max_val)
        metadata["normalized"] = True
        metadata["dtype"] = "uint8"
        metadata["value_range"] = (0, 255)
    else:
        metadata["normalized"] = False
        metadata["dtype"] = str(pixel_array.dtype)
        metadata["value_range"] = (float(np.min(pixel_array)), float(np.max(pixel_array)))

    metadata["shape"] = list(pixel_array.shape)
    metadata["dimension"] = dimension

    # 保存到文件（如果指定）
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output_path), pixel_array)
        metadata["output_path"] = str(output_path)

    # 构建响应内容
    content = [
        TextBlock(
            type="text",
            text=(
                f"DICOM数据处理完成。\n"
                f"维度: {dimension.upper()}\n"
                f"形状: {metadata['shape']}\n"
                f"数据类型: {metadata['dtype']}\n"
                f"值范围: {metadata['value_range']}\n"
                f"归一化: {metadata['normalized']}\n"
                f"患者ID: {metadata.get('PatientID', 'Unknown')}\n"
                f"模态: {metadata.get('Modality', 'Unknown')}\n"
                + (f"输出路径: {metadata.get('output_path', 'N/A')}\n" if "output_path" in metadata else "")
            ),
        )
    ]

    return ToolResponse(content=content, metadata=metadata)


def process_dicom_2d(
    dicom_path: Union[str, Path],
    normalize: bool = True,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> ToolResponse:
    """处理2D DICOM医学影像数据并归一化到(0, 255)。

    Args:
        dicom_path (Union[str, Path]): DICOM文件路径
        normalize (bool): 是否归一化到(0, 255)，默认为True
        min_val (Optional[float]): 归一化的最小值，如果为None则使用图像的最小值
        max_val (Optional[float]): 归一化的最大值，如果为None则使用图像的最大值
        output_path (Optional[Union[str, Path]]): 输出路径（.npy格式），如果为None则不保存

    Returns:
        ToolResponse: 包含处理结果的工具响应
    """
    return process_dicom_image(
        dicom_path=dicom_path,
        dimension="2d",
        normalize=normalize,
        min_val=min_val,
        max_val=max_val,
        output_path=output_path,
    )


def process_dicom_3d(
    dicom_dir: Union[str, Path],
    normalize: bool = True,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> ToolResponse:
    """处理3D DICOM医学影像数据并归一化到(0, 255)。

    Args:
        dicom_dir (Union[str, Path]): 包含DICOM文件的目录路径
        normalize (bool): 是否归一化到(0, 255)，默认为True
        min_val (Optional[float]): 归一化的最小值，如果为None则使用图像的最小值
        max_val (Optional[float]): 归一化的最大值，如果为None则使用图像的最大值
        output_path (Optional[Union[str, Path]]): 输出路径（.npy格式），如果为None则不保存

    Returns:
        ToolResponse: 包含处理结果的工具响应
    """
    return process_dicom_image(
        dicom_path=dicom_dir,
        dimension="3d",
        normalize=normalize,
        min_val=min_val,
        max_val=max_val,
        output_path=output_path,
    )
