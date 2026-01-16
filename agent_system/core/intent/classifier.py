"""
意图分类器

采用策略模式，支持多种分类策略的组合
实现规则优先、LLM 兜底的混合分类模式
"""

from typing import List, Optional, Dict, Any

from langchain_core.language_models import BaseChatModel

from .models import Intent, IntentConfig, load_intent_config
from .strategies.base import ClassifierStrategy
from .strategies.rule import RuleClassifierStrategy
from .strategies.llm import LLMClassifierStrategy


class IntentClassifier:
    """
    意图分类器
    
    使用策略模式组合多种分类策略
    支持规则优先、LLM 兜底的混合分类模式
    
    使用方式:
        >>> classifier = IntentClassifier(llm=llm, tools=tools)
        >>> intent = classifier.classify("你好")
        >>> print(intent.type)  # "chat"
    """
    
    def __init__(
        self,
        strategies: Optional[List[ClassifierStrategy]] = None,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List] = None,
        config: Optional[IntentConfig] = None,
        confidence_threshold: Optional[float] = None,
    ):
        """
        初始化分类器
        
        Args:
            strategies: 分类策略列表（按优先级排序）
                       如果不提供，将使用默认策略（规则 + LLM）
            llm: 语言模型实例（用于默认 LLM 策略）
            tools: 工具列表（用于工具名匹配）
            config: 意图配置，为 None 则从文件加载
            confidence_threshold: 置信度阈值，低于此值会尝试下一个策略
        """
        # 加载配置
        self.config = config or load_intent_config()
        self.confidence_threshold = (
            confidence_threshold 
            if confidence_threshold is not None 
            else self.config.confidence_threshold
        )
        
        # 存储通用依赖
        self.llm = llm
        self.tools = tools or []
        
        # 初始化策略
        if strategies:
            self.strategies = strategies
        else:
            self.strategies = self._create_default_strategies(llm, tools)
        
        # 按优先级排序策略
        self.strategies.sort(key=lambda s: s.priority, reverse=True)
    
    def _create_default_strategies(
        self, 
        llm: Optional[BaseChatModel],
        tools: Optional[List]
    ) -> List[ClassifierStrategy]:
        """
        创建默认策略列表
        
        默认使用规则 + LLM 混合模式
        """
        strategies = []
        
        # 规则策略（优先级高）
        rule_strategy = RuleClassifierStrategy(tools=tools)
        strategies.append(rule_strategy)
        
        # LLM 策略（优先级低，作为兜底）
        if llm:
            llm_strategy = LLMClassifierStrategy(llm=llm, tools=tools)
            strategies.append(llm_strategy)
        
        return strategies
    
    def add_strategy(self, strategy: ClassifierStrategy) -> None:
        """
        添加新的分类策略
        
        Args:
            strategy: 分类策略实例
        """
        self.strategies.append(strategy)
        # 重新排序
        self.strategies.sort(key=lambda s: s.priority, reverse=True)
    
    def remove_strategy(self, name: str) -> bool:
        """
        移除分类策略
        
        Args:
            name: 策略名称
            
        Returns:
            是否成功移除
        """
        for i, strategy in enumerate(self.strategies):
            if strategy.name == name:
                self.strategies.pop(i)
                return True
        return False
    
    def classify(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Intent:
        """
        对用户查询进行意图分类
        
        分类流程：
        1. 按优先级依次尝试每个策略
        2. 如果策略返回结果且置信度 >= 阈值，直接返回
        3. 如果置信度 < 阈值，记录结果并尝试下一个策略
        4. 如果所有策略都不满足，返回置信度最高的结果
        5. 如果没有任何结果，返回默认意图
        
        Args:
            query: 用户查询文本
            context: 上下文信息（可选）
            
        Returns:
            Intent 实例
        """
        # 构建上下文
        ctx = self._build_context(context)
        
        # 存储所有候选结果
        candidates: List[Intent] = []
        
        # 按优先级尝试每个策略
        for strategy in self.strategies:
            try:
                result = strategy.classify(query, ctx)
                
                if result is None:
                    continue
                
                # 置信度满足阈值，直接返回
                if result.confidence >= self.confidence_threshold:
                    print(f"🎯 意图分类: {result.type} "
                          f"(策略: {strategy.name}, 置信度: {result.confidence:.2f})")
                    return result
                
                # 置信度不足，记录候选
                candidates.append(result)
                
            except Exception as e:
                print(f"策略 {strategy.name} 分类失败: {e}")
                continue
        
        # 从候选中选择置信度最高的
        if candidates:
            best = max(candidates, key=lambda c: c.confidence)
            print(f"🎯 意图分类(最佳候选): {best.type} "
                  f"(策略: {best.metadata.get('strategy', 'unknown')}, "
                  f"置信度: {best.confidence:.2f})")
            return best
        
        # 没有任何结果，返回默认意图
        default_intent = Intent(
            type=self.config.default_intent,
            query=query,
            confidence=0.0,
            metadata={"strategy": "default", "reason": "no_match"}
        )
        print(f"🎯 意图分类(默认): {default_intent.type}")
        return default_intent
    
    def _build_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        构建完整的上下文
        
        合并传入的上下文和分类器持有的依赖
        """
        ctx = {
            "llm": self.llm,
            "tools": self.tools,
            "config": self.config,
        }
        
        if context:
            ctx.update(context)
        
        return ctx
    
    def set_tools(self, tools: List) -> None:
        """
        更新工具列表
        
        同时更新所有策略的工具列表
        """
        self.tools = tools
        
        for strategy in self.strategies:
            if hasattr(strategy, 'tools'):
                strategy.tools = tools
    
    def set_llm(self, llm: BaseChatModel) -> None:
        """
        更新 LLM 实例
        
        同时更新 LLM 策略
        """
        self.llm = llm
        
        for strategy in self.strategies:
            if hasattr(strategy, 'llm'):
                strategy.llm = llm


def create_classifier(
    llm: Optional[BaseChatModel] = None,
    tools: Optional[List] = None,
    use_llm: bool = True,
    confidence_threshold: float = 0.7,
) -> IntentClassifier:
    """
    工厂函数：创建意图分类器
    
    Args:
        llm: 语言模型实例
        tools: 工具列表
        use_llm: 是否使用 LLM 分类（设为 False 则仅使用规则）
        confidence_threshold: 置信度阈值
        
    Returns:
        IntentClassifier 实例
    """
    strategies = [RuleClassifierStrategy(tools=tools)]
    
    if use_llm and llm:
        strategies.append(LLMClassifierStrategy(llm=llm, tools=tools))
    
    return IntentClassifier(
        strategies=strategies,
        llm=llm,
        tools=tools,
        confidence_threshold=confidence_threshold,
    )
