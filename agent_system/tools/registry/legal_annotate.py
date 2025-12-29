"""
法律文书标注工具

用于标注 PDF 文档中的指定字段及其位置
"""

from .base import BaseTool, register_tool


@register_tool
class AnnotateLegalPdfTool(BaseTool):
    """标注法律文书 PDF"""
    
    name = "annotate_legal_pdf"
    description = """标注 PDF 法律文书，提取指定字段及其在文档中的位置。

        输入参数格式（用逗号分隔）：
        file_path,key_fields

        其中：
        - file_path: 文件路径
        - key_fields: 要提取的字段，用分号分隔，如 "案号;主办律师;法院"

        支持的字段：
        - 案号、案件编号
        - 主办律师、承办律师
        - 协办律师
        - 文件类型、案件类型
        - 当事人、原告、被告
        - 委托人
        - 法院
        - 案由
        - 开庭时间、开庭地点
        - 落款日期、落款地点、落款人

        返回：每个字段的内容和位置信息（边界框坐标）

        使用场景：需要知道信息在文档中的具体位置时使用
        示例输入：/path/to/附件九.pdf,案号;法院;当事人"""
    tags = ["legal", "annotate", "pdf"]
    
    def execute(self, input_str: str) -> str:
        """执行 PDF 标注"""
        try:
            if self.vl_tools is None:
                return "错误：VL 工具未初始化"
            
            # 解析输入参数
            parts = input_str.split(',', 1)
            if len(parts) != 2:
                return "错误：输入格式应为 'file_path,key_fields'，例如：'/path/to/file.pdf,案号;法院;当事人'"
            
            file_path = parts[0].strip()
            key_fields = parts[1].strip().replace(';', ',')
            
            result = self.vl_tools.annotate_legal_pdf(
                file_path=file_path,
                key_fields=key_fields,
                temperature=0.1
            )
            
            return str(result)
        except Exception as e:
            return f"错误：{str(e)}"

