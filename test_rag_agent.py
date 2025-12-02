"""
RAG Agent 测试脚本

测试 RAG 增强的 Agent 功能
"""

import os
from pathlib import Path
from langchain_openai import ChatOpenAI

from agent_system.config.settings import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_BASE_URL,
    LLM_SEED,
    KNOWLEDGE_BASE_DOCS_DIR
)
from agent_system.core import Agent
from agent_system.tools import Qwen25VLTools, finish_tool, create_rag_search_tool
from agent_system.rag import KnowledgeBase
from langchain_core.tools import Tool


def create_test_tools(vl_tools, knowledge_base):
    """创建测试用的工具集"""
    
    # 简化的提取关键信息工具
    extract_key_info_tool = Tool(
        name="extract_legal_key_info",
        description="提取法律文书的关键信息，输入文件路径",
        func=lambda file_path: str(vl_tools.extract_key_info(file_path=file_path))
    )
    
    # RAG 检索工具
    rag_tool = create_rag_search_tool(knowledge_base)
    
    return [extract_key_info_tool, rag_tool, finish_tool]


def test_knowledge_base():
    """测试知识库功能"""
    print("\n" + "=" * 70)
    print("测试 1: 知识库基础功能")
    print("=" * 70)
    
    try:
        # 初始化知识库
        print("\n1. 初始化知识库...")
        vl_tools = Qwen25VLTools()
        kb = KnowledgeBase(vl_tools=vl_tools)
        
        # 获取知识库信息
        info = kb.get_info()
        print(f"✓ 知识库信息:")
        print(f"  集合名称: {info['collection_name']}")
        print(f"  文档数量: {info['total_documents']}")
        print(f"  Embedding 模型: {info['embedding_model']}")
        
        if info['total_documents'] == 0:
            print("\n⚠ 警告: 知识库为空，请先运行 build_knowledge_base.py 构建知识库")
            return None
        
        # 测试检索
        print("\n2. 测试知识检索...")
        test_queries = [
            "合同纠纷",
            "法院判决",
            "案件受理"
        ]
        
        for query in test_queries:
            print(f"\n查询: {query}")
            results = kb.search(query=query, top_k=2, with_score=True)
            
            if results:
                print(f"找到 {len(results)} 条相关结果:")
                for i, result in enumerate(results, 1):
                    print(f"\n  结果 {i}:")
                    print(f"    相似度: {result['similarity']:.3f}")
                    print(f"    来源: {result['metadata'].get('file_name', '未知')}")
                    print(f"    内容预览: {result['content'][:100]}...")
            else:
                print("  未找到相关结果")
        
        print("\n✓ 知识库测试通过")
        return kb
    
    except Exception as e:
        print(f"\n✗ 知识库测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_rag_agent(knowledge_base):
    """测试 RAG 增强的 Agent"""
    print("\n" + "=" * 70)
    print("测试 2: RAG 增强的 Agent")
    print("=" * 70)
    
    if not knowledge_base:
        print("\n⚠ 跳过测试: 知识库未初始化")
        return
    
    try:
        # 初始化 LLM
        print("\n1. 初始化 LLM...")
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            base_url=LLM_BASE_URL,
            model_kwargs={"seed": LLM_SEED}
        )
        print("✓ LLM 初始化成功")
        
        # 创建工具
        print("\n2. 创建工具...")
        vl_tools = Qwen25VLTools()
        tools = create_test_tools(vl_tools, knowledge_base)
        print(f"✓ 创建了 {len(tools)} 个工具")
        
        # 创建 Agent（启用 RAG）
        print("\n3. 创建 RAG Agent...")
        agent = Agent(
            llm=llm,
            tools=tools,
            knowledge_base=knowledge_base,
            use_rag=True
        )
        print("✓ RAG Agent 创建成功")
        
        # 测试任务
        print("\n4. 测试任务执行...")
        test_task = "请告诉我关于合同纠纷案件的处理流程"
        
        print(f"\n任务: {test_task}")
        print("\n" + "-" * 70)
        
        result = agent.run(test_task)
        
        print("\n" + "-" * 70)
        print("\n最终结果:")
        print(result)
        
        print("\n✓ RAG Agent 测试通过")
    
    except Exception as e:
        print(f"\n✗ RAG Agent 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


def test_agent_without_rag(knowledge_base):
    """测试不使用 RAG 的 Agent（对比）"""
    print("\n" + "=" * 70)
    print("测试 3: 不使用 RAG 的 Agent（对比测试）")
    print("=" * 70)
    
    try:
        # 初始化 LLM
        print("\n1. 初始化 LLM...")
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            base_url=LLM_BASE_URL,
            model_kwargs={"seed": LLM_SEED}
        )
        
        # 创建工具（不包含 RAG 工具）
        vl_tools = Qwen25VLTools()
        extract_key_info_tool = Tool(
            name="extract_legal_key_info",
            description="提取法律文书的关键信息，输入文件路径",
            func=lambda file_path: str(vl_tools.extract_key_info(file_path=file_path))
        )
        tools = [extract_key_info_tool, finish_tool]
        
        # 创建 Agent（不启用 RAG）
        print("\n2. 创建普通 Agent（不使用 RAG）...")
        agent = Agent(
            llm=llm,
            tools=tools,
            knowledge_base=None,
            use_rag=False
        )
        print("✓ 普通 Agent 创建成功")
        
        # 测试任务
        print("\n3. 测试任务执行...")
        test_task = "请告诉我关于合同纠纷案件的处理流程"
        
        print(f"\n任务: {test_task}")
        print("\n" + "-" * 70)
        
        result = agent.run(test_task)
        
        print("\n" + "-" * 70)
        print("\n最终结果:")
        print(result)
        
        print("\n✓ 普通 Agent 测试通过")
        
        print("\n" + "=" * 70)
        print("对比说明:")
        print("=" * 70)
        print("RAG Agent 会在任务开始前检索相关知识，")
        print("并将这些知识作为上下文提供给 LLM，")
        print("从而提供更准确、更有针对性的回答。")
        print("=" * 70)
    
    except Exception as e:
        print(f"\n✗ 普通 Agent 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("RAG Agent 测试套件")
    print("=" * 70)
    
    # 检查环境变量
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("\n✗ 错误: 未设置 DASHSCOPE_API_KEY 环境变量")
        print("请在 .env 文件中配置 DASHSCOPE_API_KEY")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n✗ 错误: 未设置 OPENAI_API_KEY 环境变量")
        print("请在 .env 文件中配置 OPENAI_API_KEY（用于 DeepSeek）")
        return
    
    # 运行测试
    try:
        # 测试 1: 知识库功能
        kb = test_knowledge_base()
        
        # 测试 2: RAG Agent
        if kb:
            test_rag_agent(kb)
            
            # 测试 3: 对比测试
            test_agent_without_rag(kb)
        
        print("\n" + "=" * 70)
        print("所有测试完成！")
        print("=" * 70)
    
    except KeyboardInterrupt:
        print("\n\n用户中断测试")
    
    except Exception as e:
        print(f"\n✗ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



