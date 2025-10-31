"""
回调处理器

实现 LLM 输出的流式处理
"""

import sys
from typing import Any, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import GenerationChunk, ChatGenerationChunk, LLMResult


class MyPrintHandler(BaseCallbackHandler):
    """
    自定义回调处理器
    
    作用：实时打印 LLM 的输出流
    在 Agent 思考过程中，可以看到 LLM 逐字输出的思考内容
    这对于调试和理解 Agent 的推理过程非常有帮助
    
    使用方式:
        >>> handler = MyPrintHandler()
        >>> llm.invoke(prompt, config={"callbacks": [handler]})
    """
    
    def __init__(self):
        """初始化回调处理器"""
        BaseCallbackHandler.__init__(self)
    
    def on_llm_new_token(
            self,
            token: str,
            *,
            chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        """
        每生成一个新 token 时调用
        
        实现流式输出效果
        
        参数:
            token: LLM 生成的每一个 token（字符或词）
            chunk: 生成的块
            run_id: 运行 ID
            parent_run_id: 父运行 ID
        
        返回:
            token: 返回 token
        """
        end = ""
        content = token + end
        sys.stdout.write(content)  # 立即输出到终端
        sys.stdout.flush()  # 刷新缓冲区，确保立即显示
        return token
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """
        LLM 输出完成时调用
        
        添加换行符，使输出更清晰
        
        参数:
            response: LLM 的完整响应
        
        返回:
            response: 返回响应
        """
        end = ""
        content = "\n" + end
        sys.stdout.write(content)
        sys.stdout.flush()
        return response


