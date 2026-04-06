# -*- coding: utf-8 -*-
"""
Medical Data Cleaner - 医学数据清洗Agent框架

支持多种数据格式的自动识别和处理：
- 图片文件（通过OCR提取文本）
- 文本文件
- CSV结构化数据

处理流程根据数据类型自动调整：
- 图片/文本：预处理 -> 信息抽取 -> 标准化 -> 量纲统一
- CSV结构化数据：直接 -> 标准化 -> 量纲统一（跳过预处理和信息抽取）
"""

__version__ = "1.0.0"
__author__ = "Medical Agent System Team"
