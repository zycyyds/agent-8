# -*- coding: utf-8 -*-
"""
Medical Data Preprocessing Tool.
This tool handles both text and image inputs, extracting text from images if needed.
"""
import os
from typing import Union

from agentscope.message import TextBlock
from agentscope.tool._response import ToolResponse
from agentscope.tool._text_processing._medical_clean import clean_medical_text

try:
    from tools.tools_ocr import extract_text_from_image, is_image_file
except ImportError:
    from .tools_ocr import extract_text_from_image, is_image_file


def preprocess_medical_input(
    input_path_or_text: str,
    auto_ocr: bool = True,
) -> ToolResponse:
    """
    Preprocess medical input (text or image).
    If the input is an image, extract text using OCR first.
    Then clean the text using medical text cleaning tool.
    
    Args:
        input_path_or_text (str): Either a text string or a path to an image file.
        auto_ocr (bool): Whether to automatically use OCR for image files.
    
    Returns:
        ToolResponse: A tool response containing the cleaned text.
                     If input is an image, OCR is performed first.
    """
    # Check if input is a file path
    if os.path.exists(input_path_or_text) and os.path.isfile(input_path_or_text):
        # Check if it's an image file
        if is_image_file(input_path_or_text):
            if not auto_ocr:
                return ToolResponse(
                    content=[TextBlock(
                        type="text",
                        text=f"Input is an image file: {input_path_or_text}. OCR is disabled."
                    )]
                )
            
            # Extract text from image using OCR
            ocr_result = extract_text_from_image(input_path_or_text)
            
            # Check if OCR was successful
            if ocr_result.content and len(ocr_result.content) > 0:
                ocr_text = ocr_result.content[0].get("text", "")
                
                # Check for errors in OCR result
                if ocr_text.startswith("Error:") or ocr_text.startswith("Warning:"):
                    return ocr_result
                
                # Clean the extracted text
                cleaned_result = clean_medical_text(ocr_text)
                return cleaned_result
            else:
                return ocr_result
        else:
            # It's a file but not an image, try to read as text
            try:
                with open(input_path_or_text, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                # Clean the text
                return clean_medical_text(text_content)
            except Exception as e:
                return ToolResponse(
                    content=[TextBlock(
                        type="text",
                        text=f"Error reading file: {str(e)}"
                    )]
                )
    else:
        # Input is treated as text string
        # Clean the text directly
        return clean_medical_text(input_path_or_text)
