"""
LLM 分类策略

使用大语言模型进行意图分类
作为规则分类的补充，处理复杂或不确定的情况
"""

import json
from typing import Dict, Optional, Any, List

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .base import ClassifierStrategy, register_strategy
from ..models import Intent, load_intent_config


# 意图分类提示词模板
INTENT_CLASSIFICATION_PROMPT = """你是一个意图分类器。根据用户输入，判断其意图类型。

可用的意图类型：
{intent_types_desc}

可用工具列表：
{tools_desc}

用户输入：{query}

请分析用户意图，并以 JSON 格式输出：
{{"type": "意图类型名称", "confidence": 置信度(0-1之间的小数), "tool_name": "识别到的工具名(如有，否则为null)"}}

注意：
1. 只输出 JSON，不要有其他文字
2. type 必须是上述意图类型之一
3. confidence 表示你对分类结果的确信程度
4. 如果用户明确提到要使用某个工具，在 tool_name 中填写工具名
"""


@register_strategy("llm", priority=50)
class LLMClassifierStrategy(ClassifierStrategy):
    """
    LLM 分类策略
    
    使用大语言模型进行意图分类
    优先级低于规则策略，作为兜底方案
    """
    
    priority = 50
    name = "llm"
    
    def __init__(
        self, 
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List] = None,
        intent_config_path: Optional[str] = None
    ):
        """
        初始化 LLM 分类器
        
        Args:
            llm: 语言模型实例
            tools: 工具列表（用于提供给 LLM 参考）
            intent_config_path: 意图配置文件路径
        """
        self.llm = llm
        self.tools = tools or []
        self.intent_config_path = intent_config_path
        self._prompt_template = PromptTemplate.from_template(INTENT_CLASSIFICATION_PROMPT)
    
    def classify(self, query: str, context: Optional[Dict[str, Any]] = None) -> Optional[Intent]:
        """
        使用 LLM 进行意图分类
        
        Args:
            query: 用户查询
            context: 上下文信息，可包含 llm、tools 等
            
        Returns:
            Intent 实例，如果分类失败则返回 None
        """
        context = context or {}
        
        # 获取 LLM（优先使用上下文中的，其次使用实例的）
        llm = context.get("llm", self.llm)
        if llm is None:
            return None
        
        # 获取工具列表
        tools = context.get("tools", self.tools)
        
        # 加载意图配置
        intent_config = load_intent_config(self.intent_config_path)
        
        # 构建意图类型描述
        intent_types_desc = "\n".join([
            f"- {t.name}: {t.description}"
            for t in intent_config.intent_types
        ])
        
        # 构建工具描述
        if tools:
            tools_desc = "\n".join([
                f"- {getattr(t, 'name', str(t))}: {getattr(t, 'description', '无描述')[:100]}"
                for t in tools
            ])
        else:
            tools_desc = "（无可用工具）"
        
        try:
            # 构建 LLM chain
            chain = self._prompt_template | llm | StrOutputParser()
            
            # 调用 LLM
            result = chain.invoke({
                "query": query,
                "intent_types_desc": intent_types_desc,
                "tools_desc": tools_desc,
            })
            
            # 解析结果
            return self._parse_llm_response(result, query, intent_config)
            
        except Exception as e:
            print(f"LLM 意图分类失败: {e}")
            return None
    
    def _parse_llm_response(
        self, 
        response: str, 
        query: str,
        intent_config
    ) -> Optional[Intent]:
        """
        解析 LLM 响应
        
        Args:
            response: LLM 原始响应
            query: 原始查询
            intent_config: 意图配置
            
        Returns:
            Intent 实例，解析失败返回 None
        """
        try:
            # 尝试提取 JSON
            response = response.strip()
            
            # 处理可能的 markdown 代码块
            if response.startswith("```"):
                lines = response.split("\n")
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith("```") and not in_json:
                        in_json = True
                        continue
                    elif line.startswith("```") and in_json:
                        break
                    elif in_json:
                        json_lines.append(line)
                response = "\n".join(json_lines)
            
            # 解析 JSON
            data = json.loads(response)
            
            intent_type = data.get("type", "")
            confidence = float(data.get("confidence", 0.8))
            tool_name = data.get("tool_name")
            
            # 验证意图类型
            if not intent_config.is_valid_type(intent_type):
                intent_type = intent_config.default_intent
                confidence = 0.5  # 降低置信度
            
            # 构建元数据
            metadata = {
                "strategy": "llm",
                "raw_response": response[:200],  # 保留部分原始响应用于调试
            }
            if tool_name:
                metadata["tool_name"] = tool_name
            
            return Intent(
                type=intent_type,
                query=query,
                confidence=confidence,
                metadata=metadata,
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"解析 LLM 响应失败: {e}, 响应: {response[:100]}")
            return None
