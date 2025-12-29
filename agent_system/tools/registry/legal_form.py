"""
法律表单识别工具

用于识别法律案件中的各类表格表单
"""

from .base import BaseTool, register_tool


@register_tool
class RecognizeFormTool(BaseTool):
    """识别法律案件表单"""
    
    name = "recognize_form"
    description = """识别法律案件表单/表格信息。

        输入参数格式（用逗号分隔）：
        file_path,table_type

        其中：
        - file_path: 文件路径
        - table_type: 表单类型，可选值：
        * main: 案件主信息表
        * task: 任务与期限明细表
        * fee: 案件收费明细表
        * asset: 财产保全明细表
        * custom: 自定义表单（默认）

        返回：结构化的表单数据

        使用场景：识别结构化的表格信息
        示例输入：/path/to/附件九.pdf,custom"""
    tags = ["legal", "form", "table"]
    
    def execute(self, input_str: str) -> str:
        """执行表单识别"""
        try:
            if self.vl_tools is None:
                return "错误：VL 工具未初始化"
            
            # 解析输入参数
            parts = input_str.split(',')
            file_path = parts[0].strip()
            table_type = parts[1].strip() if len(parts) > 1 else "custom"
            
            result = self.vl_tools.recognize_form(
                file_path=file_path,
                table_type=table_type
            )
            
            return str(result)
        except Exception as e:
            return f"错误：{str(e)}"

