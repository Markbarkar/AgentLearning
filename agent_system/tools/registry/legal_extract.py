"""
法律文书提取工具

包含：
- 提取关键信息
- 提取文档全文
"""

from .base import BaseTool, register_tool


@register_tool
class ExtractKeyInfoTool(BaseTool):
    """提取法律文书关键信息"""
    
    name = "extract_legal_key_info"
    description = """提取法律文书的关键信息。
        
        输入参数：
        - file_path: 文件路径（必填）

        返回信息包括：
        - title: 文件标题
        - case_number: 案号
        - case_reason: 案由
        - court: 法院
        - hearing_time: 开庭时间
        - hearing_place: 开庭地点
        - case_type: 案件类型
        - lead_lawyer: 主办律师
        - client_name: 委托人
        - parties: 当事人

        使用场景：快速了解法律文书的核心信息
        示例输入：/path/to/附件九.pdf"""
    tags = ["legal", "extract", "ocr"]
    
    def execute(self, file_path: str) -> str:
        """执行关键信息提取"""
        try:
            if self.vl_tools is None:
                return "错误：VL 工具未初始化"
            
            result = self.vl_tools.extract_key_info(file_path=file_path)
            return str(result)
        except Exception as e:
            return f"错误：{str(e)}"


@register_tool
class ExtractDocumentTextTool(BaseTool):
    """提取文档全文内容"""
    
    name = "extract_document_text"
    description = """提取文档的全部文字内容（OCR）。

        输入参数：
        - file_path: 文件路径（必填）

        返回：文档每一页的文字内容列表

        使用场景：需要获取文档完整文本内容时使用
        示例输入：/path/to/附件九.pdf"""
    tags = ["legal", "extract", "ocr"]
    
    def execute(self, file_path: str) -> str:
        """执行全文提取"""
        try:
            if self.vl_tools is None:
                return "错误：VL 工具未初始化"
            
            result = self.vl_tools.extract_text(file_path=file_path)
            return str(result)
        except Exception as e:
            return f"错误：{str(e)}"

