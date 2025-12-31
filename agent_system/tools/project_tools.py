"""
项目管理工具

调用外部项目管理服务的 API
"""

from os import getenv
import dotenv
import requests

from .base import register_tool, auto_register
from .context import get_current_token, get_current_user_id

dotenv.load_dotenv()


@auto_register
class ProjectTools:
    """项目管理工具集"""
    
    def __init__(self):
        self.address = getenv("PROJECTS_SERVER")
    
    def _get_headers(self) -> dict:
        """
        获取带认证的请求头
        
        从当前请求上下文中获取 token
        """
        headers = {"Content-Type": "application/json"}
        
        token = get_current_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        return headers

    @register_tool(
        name="get_tables",
        description="""获取项目的表单。

【输入格式】
{
    "name": "get_tables",
    "args": {
         "input_str": ""
     }
}

【返回信息】
项目下的所有表单""",
    )
    def get_tables(self, input_str: str) -> str:
        """
        获取项目表单
        
        自动从上下文获取用户 token 进行认证
        """
        if not self.address:
            return "错误: PROJECTS_SERVER 环境变量未配置"
        
        url = f"{self.address}/api/tables"
        headers = self._get_headers()
        
        try:
            response = requests.get(
                url, 
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return str(response.json())
            elif response.status_code == 401:
                return "错误: 认证失败，token 无效或已过期"
            else:
                return f"错误: {response.status_code} {response.text}"
                
        except requests.exceptions.Timeout:
            return "错误: 请求超时"
        except requests.exceptions.RequestException as e:
            return f"错误: 请求失败 - {str(e)}"
        
    @register_tool(
        name="get_linked_files",
        description="""获取项目关联的文件。

【输入格式】
{
    "name": "get_linked_files",
    "args": {
        "tableId": "",
        "rowId": ""
    }
}
【返回信息】
项目关联的文件""",
    )
    def get_linked_files(self, tableId: str, rowId: str) -> str:
        """
        获取项目关联的文件
        
        自动从上下文获取用户 token 进行认证
        """
        if not self.address:
            return "错误: 未知服务器地址，请检查环境变量"
        
        url = f"{self.address}/api/tables/{tableId}/data/{rowId}/files"
        headers = self._get_headers()
        
        try:
            response = requests.get(
                url, 
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return str(response.json())
            elif response.status_code == 401:
                return "错误: 认证失败，token 无效或已过期"
            else:
                return f"错误: {response.status_code} {response.text}"
                
        except requests.exceptions.Timeout:
            return "错误: 请求超时"
        except requests.exceptions.RequestException as e:
            return f"错误: 请求失败 - {str(e)}"
        
    @register_tool(
        name="upload_linked_file",
        description="""上传项目关联的文件，将文件与项目表单中的案件进行关联

【输入格式】
{
    "name": "upload_linked_file",
    "args": {
        "tableId": "",
        "rowId": "",
        "file_path": ""
    }
}
【返回信息】
上传项目关联的文件""",
    )
    def upload_linked_file(self, tableId: str, rowId: str, file_path: str) -> str:

        import os
        from pathlib import Path
        
        if not self.address:
            return "错误: 未知服务器地址，请检查环境变量"
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return f"错误: 文件不存在 - {file_path}"
        
        url = f"{self.address}/api/tables/{tableId}/data/{rowId}/files"
        
        # 获取 headers，但移除 Content-Type（让 requests 自动设置 multipart/form-data）
        headers = self._get_headers()
        headers.pop("Content-Type", None)
        
        try:
            # 使用 form-data 上传文件
            file_name = Path(file_path).name
            with open(file_path, 'rb') as f:
                files = {
                    'file': (file_name, f, self._get_mime_type(file_path))
                }
                
                response = requests.post(
                    url, 
                    headers=headers,
                    files=files,
                    timeout=60  # 文件上传可能需要更长时间
                )
            
            if response.status_code == 200:
                return str(response.json())
            elif response.status_code == 401:
                return "错误: 认证失败，token 无效或已过期"
            else:
                return f"错误: {response.status_code} {response.text}"
                
        except requests.exceptions.Timeout:
            return "错误: 请求超时"
        except requests.exceptions.RequestException as e:
            return f"错误: 请求失败 - {str(e)}"
        except IOError as e:
            return f"错误: 文件读取失败 - {str(e)}"
    
    def _get_mime_type(self, file_path: str) -> str:
        """根据文件扩展名获取 MIME 类型"""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or 'application/octet-stream'


# 创建实例（@auto_register 自动注册）
_project_tools = ProjectTools()