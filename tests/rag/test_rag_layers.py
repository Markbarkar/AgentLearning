"""
RAG 各层综合测试

测试预处理层、向量化层、存储层、检索层的完整流程
"""

import sys
import os
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from langchain_core.documents import Document


def print_header(title: str):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str):
    """打印子标题"""
    print(f"\n--- {title} ---")


# ==================== 预处理层测试 ====================

def test_legal_chunker():
    """测试法律文档分块器"""
    print_header("预处理层: 法律文档分块器")
    
    from agent_system.rag.chunkers.legal_chunker import LegalChunker
    
    chunker = LegalChunker(max_chunk_size=800, min_chunk_size=100)
    
    # 模拟法律文本
    text = """
第一章 总则

第一条 为了加强消防工作，预防火灾和减少火灾危害，保护公民人身、财产安全，维护公共安全，根据有关法律、行政法规的基本原则，结合深圳经济特区实际，制定本条例。

第二条 本条例适用于深圳经济特区消防工作及其监督管理。

第三条 消防工作贯彻预防为主、防消结合的方针，按照政府统一领导、部门依法监管、单位全面负责、公民积极参与的原则，实行消防安全责任制。

第二章 火灾预防

第四条 市、区人民政府应当将消防工作纳入国民经济和社会发展规划，将消防公共服务纳入政府公共服务体系。

第五条 消防安全委员会由本级人民政府主要负责人或者分管负责人担任主任委员，负责统筹协调、指导本行政区域消防工作。
"""
    
    metadata = {
        "law_name": "深圳经济特区消防条例",
        "region": "深圳",
        "law_type": "地方法规"
    }
    
    chunks = chunker.chunk(text, metadata)
    
    print(f"✓ 分块数量: {len(chunks)}")
    
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n  分块 {i+1}:")
        print(f"    章节: {chunk.metadata.get('chapter', 'N/A')}")
        print(f"    条款: {chunk.metadata.get('article_num', 'N/A')}")
        print(f"    内容: {chunk.page_content[:100]}...")
    
    assert len(chunks) >= 2, "应至少生成2个分块"
    print("\n✓ 法律文档分块器测试通过")
    return True


def test_metadata_extractor():
    """测试元数据提取器"""
    print_header("预处理层: 元数据提取器")
    
    from agent_system.rag.extractors.legal_metadata import LegalMetadataExtractor
    
    extractor = LegalMetadataExtractor()
    
    text = """
深圳经济特区消防条例

（2023年6月28日深圳市第七届人民代表大会常务委员会第十九次会议通过）

第一条 为了加强消防工作，预防火灾和减少火灾危害...

本条例自2023年9月1日起施行。
"""
    
    metadata = extractor.extract("深圳经济特区消防条例.docx", text)
    
    print(f"✓ 法规名称: {metadata['law_name']}")
    print(f"✓ 法规类型: {metadata['law_type']}")
    print(f"✓ 地区: {metadata['region']}")
    print(f"✓ 地区级别: {metadata['region_level']}")
    print(f"✓ 颁布日期: {metadata['publish_date']}")
    print(f"✓ 生效日期: {metadata['effective_date']}")
    
    assert metadata['law_name'] == "深圳经济特区消防条例"
    assert metadata['region'] == "深圳"
    print("\n✓ 元数据提取器测试通过")
    return True


# ==================== 向量化层测试 ====================

def test_embedding_service():
    """测试嵌入服务"""
    print_header("向量化层: 嵌入服务")
    
    from agent_system.rag.embedding import EmbeddingService
    from agent_system.config.settings import QWEN_EMBEDDING_API_KEY
    
    if not QWEN_EMBEDDING_API_KEY:
        print("⚠️ DASHSCOPE_API_KEY 未设置，跳过嵌入服务测试")
        return True
    
    service = EmbeddingService()
    
    texts = ["这是一个测试句子", "法律法规知识库"]
    embeddings = service.embed_documents(texts)
    
    print(f"✓ 嵌入维度: {len(embeddings[0])}")
    print(f"✓ 文本数量: {len(embeddings)}")
    
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1024
    print("\n✓ 嵌入服务测试通过")
    return True


def test_embedding_cache():
    """测试嵌入缓存"""
    print_header("向量化层: 嵌入缓存")
    
    from agent_system.rag.embedding import EmbeddingCache
    
    cache = EmbeddingCache()
    
    # 测试缓存
    test_text = "测试缓存文本"
    test_embedding = [0.1, 0.2, 0.3]
    
    # 初始应该未命中
    result = cache.get(test_text)
    assert result is None, "初始应该未命中"
    print("✓ 初始缓存未命中")
    
    # 设置缓存
    cache.set(test_text, test_embedding)
    
    # 应该命中
    result = cache.get(test_text)
    assert result == test_embedding, "应该命中缓存"
    print("✓ 缓存命中成功")
    
    # 清理
    cache.clear()
    print("\n✓ 嵌入缓存测试通过")
    return True


# ==================== 存储层测试 ====================

def test_bm25_index():
    """测试 BM25 索引"""
    print_header("存储层: BM25 索引")
    
    from agent_system.rag.stores.bm25_index import BM25Index
    
    index = BM25Index(
        user_id="test_user",
        collection_prefix="test_bm25",
        persist_directory="./data/test_bm25"
    )
    
    # 清空
    index.clear()
    
    # 添加文档
    docs = [
        Document(page_content="民法典是新中国第一部以法典命名的法律", metadata={"type": "法律"}),
        Document(page_content="消防法规定了火灾预防的相关内容", metadata={"type": "法律"}),
        Document(page_content="深圳经济特区消防条例是地方性法规", metadata={"type": "地方法规"}),
    ]
    
    index.add_documents(docs)
    print(f"✓ 添加 {index.get_count()} 个文档")
    
    # 搜索测试
    results = index.search("民法典", k=2)
    print(f"✓ 搜索 '民法典' 返回 {len(results)} 个结果")
    if results:
        print(f"  Top 1: {results[0].content[:50]}... (score: {results[0].score:.2f})")
    
    # 清理
    index.clear()
    print("\n✓ BM25 索引测试通过")
    return True


def test_chroma_store():
    """测试 ChromaDB 存储"""
    print_header("存储层: ChromaDB 存储")
    
    try:
        from agent_system.rag.stores.chroma_store import ChromaStore, CHROMA_AVAILABLE
        
        if not CHROMA_AVAILABLE:
            print("⚠️ ChromaDB 未安装，跳过测试")
            return True
        
        store = ChromaStore(
            user_id="test_chroma",
            collection_prefix="test_chroma",
            persist_directory="./data/test_chroma"
        )
        
        # 清空
        store.clear_collection()
        
        # 添加文档
        docs = [
            Document(page_content="测试文档1", metadata={"category": "A"}),
            Document(page_content="测试文档2", metadata={"category": "B"}),
        ]
        embeddings = [[0.1] * 1024, [0.2] * 1024]
        
        store.add_documents(docs, embeddings)
        print(f"✓ 添加 {store.get_collection_count()} 个文档")
        
        # 搜索
        results = store.similarity_search([0.15] * 1024, k=2)
        print(f"✓ 相似度搜索返回 {len(results)} 个结果")
        
        # 清理
        store.clear_collection()
        print("\n✓ ChromaDB 存储测试通过")
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB 测试失败: {e}")
        return False


def test_milvus_store():
    """测试 Milvus 存储"""
    print_header("存储层: Milvus 存储")
    
    try:
        from agent_system.rag.stores.milvus_store import MilvusStore, MILVUS_AVAILABLE
        
        if not MILVUS_AVAILABLE:
            print("⚠️ pymilvus 未安装，跳过测试")
            return True
        
        try:
            store = MilvusStore(
                user_id="test_milvus",
                collection_prefix="test_milvus",
                host="localhost",
                port=19530
            )
        except ConnectionError as e:
            print(f"⚠️ 无法连接 Milvus: {e}")
            return True
        
        # 清空
        store.clear_collection()
        
        # 添加文档
        docs = [
            Document(
                page_content="中华人民共和国民法典第一条",
                metadata={
                    "law_name": "中华人民共和国民法典",
                    "law_type": "法律",
                    "region": "全国",
                    "article_num": "第一条",
                }
            ),
            Document(
                page_content="深圳经济特区消防条例第一条",
                metadata={
                    "law_name": "深圳经济特区消防条例",
                    "law_type": "地方法规",
                    "region": "深圳",
                    "article_num": "第一条",
                }
            ),
        ]
        embeddings = [[0.1] * 1024, [0.2] * 1024]
        
        store.add_documents(docs, embeddings)
        
        import time
        time.sleep(1)  # 等待数据刷新
        
        print(f"✓ 添加 {store.get_collection_count()} 个文档")
        
        # 搜索
        results = store.similarity_search([0.15] * 1024, k=2)
        print(f"✓ 相似度搜索返回 {len(results)} 个结果")
        
        # 按条件搜索
        results_filtered = store.similarity_search(
            [0.15] * 1024, k=2, filter={"region": "深圳"}
        )
        print(f"✓ 按地区过滤返回 {len(results_filtered)} 个结果")
        
        # 清理
        store.clear_collection()
        store.close()
        print("\n✓ Milvus 存储测试通过")
        return True
        
    except Exception as e:
        print(f"❌ Milvus 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==================== 检索层测试 ====================

def test_query_parser():
    """测试查询解析器"""
    print_header("检索层: 查询解析器")
    
    from agent_system.rag.retrieval import QueryParser
    
    parser = QueryParser()
    
    # 测试各种查询
    queries = [
        "深圳消防条例第一条",
        "《民法典》关于合同的规定有哪些？",
        "北京市的劳动法规",
        "如何处理侵权纠纷",
    ]
    
    for query in queries:
        parsed = parser.parse(query)
        print(f"\n查询: {query}")
        print(f"  类型: {parsed.query_type.value}")
        print(f"  关键词: {parsed.keywords[:5]}")
        print(f"  实体: {parsed.entities}")
        print(f"  过滤条件: {parsed.filters}")
        if parsed.expansion_terms:
            print(f"  扩展词: {parsed.expansion_terms}")
    
    print("\n✓ 查询解析器测试通过")
    return True


def test_hybrid_retriever():
    """测试混合检索器"""
    print_header("检索层: 混合检索器")
    
    from agent_system.rag.retrieval import HybridRetriever, RetrievalResult
    from agent_system.rag.stores.bm25_index import BM25Index
    
    # 创建 BM25 索引并添加测试数据
    bm25_index = BM25Index(
        user_id="test_hybrid",
        collection_prefix="test_hybrid_bm25",
        persist_directory="./data/test_hybrid_bm25"
    )
    bm25_index.clear()
    
    docs = [
        Document(page_content="民法典规定了公民的基本权利", metadata={"type": "法律"}),
        Document(page_content="消防安全是城市管理的重点", metadata={"type": "法规"}),
        Document(page_content="劳动合同法保护劳动者权益", metadata={"type": "法律"}),
    ]
    bm25_index.add_documents(docs)
    
    # 测试只使用 BM25 的混合检索
    retriever = HybridRetriever(
        bm25_index=bm25_index,
        vector_store=None,
        embedding_service=None
    )
    
    results = retriever.retrieve("民法典", k=2, use_vector=False, use_bm25=True)
    
    print(f"✓ 检索 '民法典' 返回 {len(results)} 个结果")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r.content[:40]}... (score: {r.score:.3f})")
    
    # 清理
    bm25_index.clear()
    print("\n✓ 混合检索器测试通过")
    return True


def test_reranker():
    """测试重排序器"""
    print_header("检索层: 重排序器")
    
    from agent_system.rag.retrieval import Reranker, RerankerType, create_legal_reranker
    from agent_system.rag.retrieval.hybrid_retriever import RetrievalResult
    
    # 创建法律专用重排序器
    reranker = create_legal_reranker()
    
    # 模拟检索结果
    results = [
        RetrievalResult(
            document=Document(
                page_content="深圳经济特区消防条例第一条",
                metadata={"region": "深圳", "article_num": "第一条", "effective_date": "2023-09-01"}
            ),
            score=0.7,
            doc_id="1"
        ),
        RetrievalResult(
            document=Document(
                page_content="消防法第一条",
                metadata={"region": "全国", "article_num": "第一条", "effective_date": "2019-01-01"}
            ),
            score=0.8,
            doc_id="2"
        ),
    ]
    
    reranked = reranker.rerank("深圳消防第一条", results, top_k=2)
    
    print(f"✓ 重排序后结果:")
    for i, r in enumerate(reranked):
        print(f"  {i+1}. {r.content[:30]}... (score: {r.score:.3f})")
    
    print("\n✓ 重排序器测试通过")
    return True


def test_retrieval_pipeline():
    """测试检索管道"""
    print_header("检索层: 检索管道")
    
    from agent_system.rag.retrieval import RetrievalPipeline, RetrievalConfig
    from agent_system.rag.stores.bm25_index import BM25Index
    
    # 创建 BM25 索引
    bm25_index = BM25Index(
        user_id="test_pipeline",
        collection_prefix="test_pipeline_bm25",
        persist_directory="./data/test_pipeline_bm25"
    )
    bm25_index.clear()
    
    # 添加测试文档
    docs = [
        Document(
            page_content="深圳经济特区消防条例第一条 为了加强消防工作",
            metadata={"law_name": "深圳经济特区消防条例", "region": "深圳", "article_num": "第一条"}
        ),
        Document(
            page_content="消防法第二条 火灾预防是消防工作的重点",
            metadata={"law_name": "消防法", "region": "全国", "article_num": "第二条"}
        ),
        Document(
            page_content="民法典第一条 保护民事主体的合法权益",
            metadata={"law_name": "民法典", "region": "全国", "article_num": "第一条"}
        ),
    ]
    bm25_index.add_documents(docs)
    
    # 创建检索管道（只使用 BM25，因为没有向量存储）
    config = RetrievalConfig(
        use_vector=False,
        use_bm25=True,
        enable_rerank=True,
        top_k=3
    )
    
    pipeline = RetrievalPipeline(
        bm25_index=bm25_index,
        config=config
    )
    
    # 测试检索
    query = "深圳消防条例第一条"
    result = pipeline.search_with_context(query)
    
    print(f"查询: {query}")
    print(f"解析结果: {result['parsed_query']}")
    print(f"返回结果: {len(result['results'])} 个")
    
    for i, r in enumerate(result['results']):
        print(f"  {i+1}. {r.content[:40]}... (score: {r.score:.3f})")
    
    # 测试格式化输出
    formatted = pipeline.format_results_for_llm(result['results'])
    print(f"\n格式化输出 (前200字符):\n{formatted[:200]}...")
    
    # 清理
    bm25_index.clear()
    print("\n✓ 检索管道测试通过")
    return True


# ==================== 主测试函数 ====================

def run_all_tests():
    """运行所有测试"""
    print("\n")
    print("=" * 70)
    print("           RAG 各层综合测试")
    print("=" * 70)
    
    tests = [
        # 预处理层
        ("预处理层 - 法律分块器", test_legal_chunker),
        ("预处理层 - 元数据提取器", test_metadata_extractor),
        
        # 向量化层
        ("向量化层 - 嵌入服务", test_embedding_service),
        ("向量化层 - 嵌入缓存", test_embedding_cache),
        
        # 存储层
        ("存储层 - BM25 索引", test_bm25_index),
        ("存储层 - ChromaDB", test_chroma_store),
        ("存储层 - Milvus", test_milvus_store),
        
        # 检索层
        ("检索层 - 查询解析器", test_query_parser),
        ("检索层 - 混合检索器", test_hybrid_retriever),
        ("检索层 - 重排序器", test_reranker),
        ("检索层 - 检索管道", test_retrieval_pipeline),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ 测试失败: {name}")
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n")
    print("=" * 70)
    print(f"  测试完成: {passed} 通过, {failed} 失败")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
