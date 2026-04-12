# -*- coding: utf-8 -*-
"""
Tools module - 数据处理工具集

导出主要工具函数供外部使用。
"""

from tools.data_type_detector import (
    detect_data_type,
    detect_input_type,
    get_processing_pipeline,
    format_detection_result
)

from tools.csv_reader import (
    read_csv_columns,
    read_csv_sample,
    read_csv_data,
    analyze_columns_for_standardization,
    get_csv_info_for_agent
)

# 标准化工具
from tools.tools_standardization import (
    standardize_term,
    normalize_unit,
    MOCK_KB,
    UNIT_CONVERSION_RULES
)

# OCR工具 (可选，需要cv2)
try:
    from tools.tools_ocr import (
        extract_text_from_image,
        is_image_file,
        extract_ocr_with_layout
    )
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    extract_text_from_image = None
    is_image_file = None
    extract_ocr_with_layout = None

# 版面分析工具 (可选，需要cv2)
try:
    from tools.tools_layout_analysis import (
        is_figure_path,
        analyze_layout,
        analyze_medical_figure,
        batch_analyze_figures,
        layout_analysis_tool
    )
    LAYOUT_ANALYSIS_AVAILABLE = True
except ImportError:
    LAYOUT_ANALYSIS_AVAILABLE = False
    is_figure_path = None
    analyze_layout = None
    analyze_medical_figure = None
    batch_analyze_figures = None
    layout_analysis_tool = None

# 预处理工具
try:
    from tools.tools_preprocess import preprocess_medical_input
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False
    preprocess_medical_input = None

__all__ = [
    # 数据类型检测
    "detect_data_type",
    "detect_input_type",
    "get_processing_pipeline",
    "format_detection_result",
    
    # CSV工具
    "read_csv_columns",
    "read_csv_sample",
    "read_csv_data",
    "analyze_columns_for_standardization",
    "get_csv_info_for_agent",
    
    # 标准化
    "standardize_term",
    "normalize_unit",
    "MOCK_KB",
    "UNIT_CONVERSION_RULES",
    
    # OCR (可选)
    "extract_text_from_image",
    "is_image_file",
    "extract_ocr_with_layout",
    "OCR_AVAILABLE",
    
    # 版面分析 (可选)
    "is_figure_path",
    "analyze_layout",
    "analyze_medical_figure",
    "batch_analyze_figures",
    "layout_analysis_tool",
    "LAYOUT_ANALYSIS_AVAILABLE",
    
    # 预处理 (可选)
    "preprocess_medical_input",
    "PREPROCESS_AVAILABLE",
]
