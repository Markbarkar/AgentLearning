"""
元数据提取器抽象基类

定义元数据提取的标准接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class ExtractorBase(ABC):
    """
    元数据提取器抽象基类
    
    所有元数据提取器都应继承此类并实现 extract 方法
    """
    
    @abstractmethod
    def extract(
        self,
        file_path: str,
        text: str
    ) -> Dict[str, Any]:
        """
        从文档提取元数据
        
        Args:
            file_path: 文件路径（用于提取文件名等信息）
            text: 文档内容
            
        Returns:
            元数据字典
        """
        pass
    
    def extract_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        从文件提取元数据（自动读取文件内容）
        
        Args:
            file_path: 文件路径
            
        Returns:
            元数据字典
        """
        # 尝试多种编码读取文件
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        text = ""
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                break
            except UnicodeDecodeError:
                continue
            except Exception:
                break
        
        return self.extract(file_path, text)
