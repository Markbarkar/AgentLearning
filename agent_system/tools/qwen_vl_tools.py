"""
Qwen2.5-VL 信息提取工具集
供 Agent 使用的工具封装

使用 @register_tool 装饰器自动注册为 LangChain Tool
"""

import requests
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from .base import register_tool, auto_register, _parse_annotate_input, _parse_form_input


@auto_register
class Qwen25VLTools:
    """Qwen2.5-VL 信息提取工具集"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.model_url = base_url + "/model"
        self.api_url = base_url + "/api"
    
    def annotate_legal_document(
        self,
        file_path: str,
        key_fields: Union[List[str], str],
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """
        标注单页法律文书图片
        
        Args:
            file_path: 法律文书图片文件路径
            key_fields: 需要提取的关键字段（列表或逗号分隔字符串）
            temperature: 采样温度
            max_tokens: 最大生成 token 数
        
        Returns:
            标注结果，包含 image_id, annotations, processing_time
        """
        url = f"{self.model_url}/annotate_legal_document"
        
        if isinstance(key_fields, list):
            key_fields_str = ",".join(key_fields)
        else:
            key_fields_str = key_fields
        
        files = {"file": open(file_path, "rb")}
        data = {
            "key_fields": key_fields_str,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
        finally:
            files["file"].close()
    
    @register_tool(
        name="annotate_legal_pdf",
        description="""标注 PDF 法律文书，提取指定字段及其在文档中的位置。

输入格式：file_path,key_fields
- file_path: PDF 文件路径
- key_fields: 要提取的字段，用分号分隔，如 "案号;法院;当事人"

支持的字段：案号、主办律师、协办律师、当事人、原告、被告、委托人、法院、案由、开庭时间等

示例：/path/to/file.pdf,案号;法院;当事人""",
        input_parser=_parse_annotate_input
    )
    def annotate_legal_pdf(
        self,
        file_path: str,
        key_fields: Union[List[str], str] = "",
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """
        标注 PDF 法律文书（多页）
        
        Args:
            file_path: PDF 文件路径
            key_fields: 需要提取的关键字段
            temperature: 采样温度
            max_tokens: 最大生成 token 数
        
        Returns:
            标注结果，包含 total_pages, all_annotations
        """
        url = f"{self.model_url}/annotate_legal_pdf"
        
        if isinstance(key_fields, list):
            key_fields_str = ",".join(key_fields)
        else:
            key_fields_str = key_fields.replace(';', ',')
        
        files = {"file": open(file_path, "rb")}
        data = {
            "key_fields": key_fields_str,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
        finally:
            files["file"].close()
    
    @register_tool(
        name="recognize_form",
        description="""识别表单/表格内容。

输入格式：file_path,table_type
- file_path: 表单图片或PDF文件路径
- table_type: 表单类型（可选），可选值：main/task/fee/asset/custom，默认 main

示例：/path/to/form.pdf,main""",
        input_parser=_parse_form_input
    )
    def recognize_form(
        self,
        file_path: str,
        table_type: str = "main",
        table_name: str = "",
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        识别表单
        
        Args:
            file_path: 文件路径
            table_type: 表单类型 (main/task/fee/asset/custom)
            table_name: 表单名称
            temperature: 采样温度
        
        Returns:
            识别结果
        """
        url = f"{self.model_url}/process_image_direct"
        
        files = {"image": open(file_path, "rb")}
        data = {
            "table_type": table_type,
            "table_name": table_name,
            "temperature": temperature
        }
        
        try:
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
        finally:
            files["image"].close()
    
    @register_tool(
        name="extract_pdf_text",
        description="""提取本地pdf或图片的全文内容（OCR），仅支持 PDF 和图片！

输入：文件路径
输出：文档的完整文本内容

示例：/path/to/document.pdf"""
    )
    def extract_text(
        self,
        file_path: Optional[str] = None,
        url: Optional[str] = None,
        temperature: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        提取本地pdf的全文（OCR）
        
        Args:
            file_path: 文件路径
            url: 文件 URL
        
        Returns:
            页面列表
        """
        api_url = f"{self.model_url}/process_file"

        print(f"api_url: {api_url}")
        
        try:
            if file_path:
                files = {"file": open(file_path, "rb")}
                response = requests.post(api_url, files=files, data={"temperature": temperature})
                files["file"].close()
            elif url:
                response = requests.post(api_url, data={"url": url, "temperature": temperature})
            else:
                return [{"error": "必须提供 file_path 或 url"}]
            
            response.raise_for_status()
            result = response.json()
            return result.get("pages", [])
        except Exception as e:
            return [{"error": str(e)}]
    
    @register_tool(
        name="extract_legal_key_info",
        description="""提取法律文书的关键信息，如案号、当事人、法院、案由、开庭时间等。

输入：法律文书文件路径（PDF 或图片）
输出：结构化的关键信息字典

适用于：起诉状、判决书、裁定书、律师函等法律文书

示例：/path/to/legal_document.pdf"""
    )
    def extract_key_info(
        self,
        file_path: Optional[str] = None,
        url: Optional[str] = None,
        temperature: float = 0.7
    ) -> Dict[str, str]:
        """
        提取法务关键信息
        
        Args:
            file_path: 文件路径
            url: 文件 URL
        
        Returns:
            关键信息字典
        """
        api_url = f"{self.model_url}/extract_key_info"
        
        try:
            if file_path:
                files = {"file": open(file_path, "rb")}
                response = requests.post(api_url, files=files, data={"temperature": temperature})
                files["file"].close()
            elif url:
                response = requests.post(api_url, data={"url": url, "temperature": temperature})
            else:
                return {"error": "必须提供 file_path 或 url"}
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def fill_form(
        self,
        ocr_content: List[Dict[str, Any]],
        form_template: Dict[str, Any],
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        表单自动填写
        
        Args:
            ocr_content: OCR 提取的内容列表
            form_template: 表单模板
            temperature: 采样温度
        
        Returns:
            填写后的表单数据
        """
        url = f"{self.model_url}/fill_form"
        
        data = {
            "ocr_content": ocr_content,
            "form_template": form_template,
            "temperature": temperature
        }
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    tools = Qwen25VLTools()
    result = tools.extract_text(file_path="data/knowledge_base/temp_uploads/附件二.pdf")
    print(result)
