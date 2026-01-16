from agent_system.core.intent import IntentClassifier
from agent_system.core.intent.strategies import RuleClassifierStrategy, LLMClassifierStrategy
from agent_system.config.settings import LLM_MODEL, LLM_BASE_URL, LLM_API_KEY

# 创建仅规则分类器
classifier = IntentClassifier(
    strategies=[RuleClassifierStrategy()],
    llm=None,
    tools=[],
)

llm_classifier = IntentClassifier(
    strategies=[LLMClassifierStrategy(
        intent_config_path="agent_system/core/intent/config/intent_config.json",
    )],
    llm=LLM_MODEL,
    tools=[],
)

# 测试不同类型的查询
test_queries = [
    '你好',
    '什么是劳动合同',
    '帮我分析这份合同并生成报告',
    '你能做什么',
]

print('意图分类测试:')
print('-' * 50)
for query in test_queries:
    intent = classifier.classify(query)
    print(f'查询: {query}')
    print(f'  -> 类型: {intent.type}, 置信度: {intent.confidence:.2f}')
    print()

print('-' * 50)
for query in test_queries:
    intent = llm_classifier.classify(query)
    print(f'查询: {query}')
    print(f'  -> 类型: {intent.type}, 置信度: {intent.confidence:.2f}')
    print()