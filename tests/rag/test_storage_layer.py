"""
存储层测试

测试向量存储和 BM25 索引功能
"""

import sys
import os
from pathlib import Path
import uuid

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# 设置环境变量以便加载配置
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from langchain_core.documents import Document

# 直接导入配置，避免触发 agent_system.rag 的全量导入
sys.path.insert(0, str(project_root / "agent_system"))
from config.settings import ENABLE_USER_ISOLATION


def test_user_isolation():
    """测试用户隔离逻辑"""
    print("\n" + "=" * 60)
    print("测试 1: 用户隔离逻辑")
    print("=" * 60)
    
    from agent_system.rag.stores.user_isolation import get_collection_name, validate_user_id
    
    # 测试 user_id 清理
    assert validate_user_id(None) is None
    assert validate_user_id("") is None
    assert validate_user_id("user123") == "user123"
    assert validate_user_id("user-123") == "user123"  # 去除非法字符
    assert validate_user_id("user@123") == "user123"
    assert validate_user_id("a" * 50) == "a" * 32  # 限制长度
    print("✓ user_id 清理测试通过")
    
    # 测试 collection 名称生成
    assert get_collection_name(None, "legal_kb", True) == "legal_kb_public"
    assert get_collection_name("user123", "legal_kb", True) == "legal_kb_user_user123"
    assert get_collection_name("user123", "legal_kb", False) == "legal_kb_public"  # 禁用隔离
    print("✓ collection 名称生成测试通过")
    
    print("\n✓ 用户隔离逻辑测试通过")


def test_bm25_index():
    """测试 BM25 索引"""
    print("\n" + "=" * 60)
    print("测试 2: BM25 索引")
    print("=" * 60)
    
    from agent_system.rag.stores.bm25_index import BM25Index
    
    # 创建测试索引
    index = BM25Index(
        user_id="test_user",
        collection_prefix="test_bm25",
        enable_isolation=True,
        persist_directory="./data/test_bm25"
    )
    
    # 清空
    index.clear()
    assert index.get_count() == 0
    print("✓ 创建和清空索引成功")
    
    # 添加测试文档
    docs = [
        Document(
            page_content="中华人民共和国民法典是新中国成立以来第一部以法典命名的法律",
            metadata={"law_name": "民法典", "type": "法律"}
        ),
        Document(
            page_content="消防法规定了火灾预防和消防救援的相关内容",
            metadata={"law_name": "消防法", "type": "法律"}
        ),
        Document(
            page_content="深圳经济特区消防条例是深圳市的地方性法规",
            metadata={"law_name": "深圳经济特区消防条例", "type": "地方法规"}
        ),
    ]
    
    ids = index.add_documents(docs)
    assert len(ids) == 3
    assert index.get_count() == 3
    print(f"✓ 添加 {len(ids)} 个文档成功")
    
    # 测试搜索
    results = index.search("民法典", k=2)
    assert len(results) > 0
    assert "民法典" in results[0].content
    print(f"✓ 搜索 '民法典' 返回 {len(results)} 个结果")
    print(f"  Top 1: {results[0].content[:50]}... (score: {results[0].score:.2f})")
    
    # 测试带过滤的搜索
    results_filtered = index.search("消防", k=5, filter={"type": "地方法规"})
    assert len(results_filtered) > 0
    assert results_filtered[0].metadata.get("type") == "地方法规"
    print(f"✓ 带过滤搜索返回 {len(results_filtered)} 个结果")
    
    # 测试删除
    deleted = index.delete_documents([ids[0]])
    assert deleted == 1
    assert index.get_count() == 2
    print("✓ 删除文档成功")
    
    # 清理
    index.clear()
    print("\n✓ BM25 索引测试通过")


def test_chroma_store():
    """测试 ChromaDB 存储"""
    print("\n" + "=" * 60)
    print("测试 3: ChromaDB 存储")
    print("=" * 60)
    
    try:
        from agent_system.rag.stores.chroma_store import ChromaStore, CHROMA_AVAILABLE
        
        if not CHROMA_AVAILABLE:
            print("⚠️ ChromaDB 未安装，跳过测试")
            return
        
        # 创建测试存储
        store = ChromaStore(
            user_id="test_chroma_user",
            collection_prefix="test_chroma",
            enable_isolation=True,
            persist_directory="./data/test_chroma"
        )
        
        # 清空
        store.clear_collection()
        assert store.get_collection_count() == 0
        print("✓ 创建和清空 ChromaDB 存储成功")
        
        # 添加测试文档
        docs = [
            Document(
                page_content="这是测试文档一",
                metadata={"source": "test1.txt", "category": "A"}
            ),
            Document(
                page_content="这是测试文档二",
                metadata={"source": "test2.txt", "category": "B"}
            ),
        ]
        
        # 模拟向量（1024维）
        embeddings = [[0.1] * 1024, [0.2] * 1024]
        
        ids = store.add_documents(docs, embeddings)
        assert len(ids) == 2
        assert store.get_collection_count() == 2
        print(f"✓ 添加 {len(ids)} 个文档成功")
        
        # 测试搜索
        query_embedding = [0.15] * 1024
        results = store.similarity_search(query_embedding, k=2)
        assert len(results) == 2
        print(f"✓ 相似度搜索返回 {len(results)} 个结果")
        for i, r in enumerate(results):
            print(f"  {i+1}. {r.content} (score: {r.score:.4f})")
        
        # 测试带过滤的搜索
        results_filtered = store.similarity_search(
            query_embedding, 
            k=2, 
            filter={"category": "A"}
        )
        assert len(results_filtered) == 1
        print(f"✓ 带过滤搜索返回 {len(results_filtered)} 个结果")
        
        # 清理
        store.clear_collection()
        print("\n✓ ChromaDB 存储测试通过")
        
    except Exception as e:
        print(f"❌ ChromaDB 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_milvus_store():
    """测试 Milvus 存储"""
    print("\n" + "=" * 60)
    print("测试 4: Milvus 存储")
    print("=" * 60)
    
    try:
        from agent_system.rag.stores.milvus_store import MilvusStore, MILVUS_AVAILABLE
        
        if not MILVUS_AVAILABLE:
            print("⚠️ pymilvus 未安装，跳过测试")
            return
        
        # 尝试连接 Milvus
        try:
            store = MilvusStore(
                user_id="test_milvus_user",
                collection_prefix="test_milvus",
                enable_isolation=True,
                host="localhost",
                port=19530
            )
        except ConnectionError as e:
            print(f"⚠️ 无法连接 Milvus: {e}")
            print("  请确保 Milvus 服务已启动")
            return
        
        # 清空
        store.clear_collection()
        assert store.get_collection_count() == 0
        print("✓ 创建和清空 Milvus 存储成功")
        
        # 添加测试文档
        docs = [
            Document(
                page_content="中华人民共和国民法典第一条",
                metadata={
                    "source": "/path/to/民法典.docx",
                    "file_name": "民法典.docx",
                    "law_name": "中华人民共和国民法典",
                    "law_type": "法律",
                    "region": "全国",
                    "chapter": "第一编 总则",
                    "article_num": "第一条",
                }
            ),
            Document(
                page_content="深圳经济特区消防条例第一条",
                metadata={
                    "source": "/path/to/深圳消防条例.docx",
                    "file_name": "深圳消防条例.docx",
                    "law_name": "深圳经济特区消防条例",
                    "law_type": "地方法规",
                    "region": "深圳",
                    "chapter": "第一章 总则",
                    "article_num": "第一条",
                }
            ),
            Document(
                page_content="深圳经济特区消防条例第二条 本条例适用于深圳经济特区",
                metadata={
                    "source": "/path/to/深圳消防条例.docx",
                    "file_name": "深圳消防条例.docx",
                    "law_name": "深圳经济特区消防条例",
                    "law_type": "地方法规",
                    "region": "深圳",
                    "chapter": "第一章 总则",
                    "article_num": "第二条",
                }
            ),
        ]
        
        # 模拟向量（1024维）
        embeddings = [
            [0.1 + i * 0.01] * 1024 for i in range(len(docs))
        ]
        
        ids = store.add_documents(docs, embeddings)
        assert len(ids) == 3
        print(f"✓ 添加 {len(ids)} 个文档成功")
        
        # 等待数据刷新
        import time
        time.sleep(1)
        
        count = store.get_collection_count()
        print(f"  文档总数: {count}")
        
        # 测试搜索
        query_embedding = [0.12] * 1024
        results = store.similarity_search(query_embedding, k=3)
        assert len(results) > 0
        print(f"✓ 相似度搜索返回 {len(results)} 个结果")
        for i, r in enumerate(results):
            print(f"  {i+1}. {r.content[:40]}... (score: {r.score:.4f})")
        
        # 测试带过滤的搜索
        results_filtered = store.similarity_search(
            query_embedding, 
            k=3, 
            filter={"region": "深圳"}
        )
        print(f"✓ 按地区过滤搜索返回 {len(results_filtered)} 个结果")
        for r in results_filtered:
            assert r.metadata.get("region") == "深圳"
        
        # 测试获取唯一文件列表
        files = store.list_unique_files()
        print(f"✓ 唯一文件列表: {len(files)} 个")
        for f in files:
            print(f"  - {f['file_name']}: {f['chunk_count']} 块")
        
        # 测试按条件删除
        deleted = store.delete_by_filter({"law_name": "深圳经济特区消防条例"})
        print(f"✓ 按条件删除: {deleted} 个文档")
        
        # 清理
        store.clear_collection()
        store.close()
        print("\n✓ Milvus 存储测试通过")
        
    except Exception as e:
        print(f"❌ Milvus 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_hybrid_store():
    """测试混合存储"""
    print("\n" + "=" * 60)
    print("测试 5: 混合存储 (HybridStore)")
    print("=" * 60)
    
    try:
        from agent_system.rag.stores.factory import HybridStore, StoreProvider
        
        # 创建混合存储（使用 ChromaDB 作为后端）
        hybrid = HybridStore.create(
            provider=StoreProvider.CHROMA,
            user_id="test_hybrid_user",
            collection_prefix="test_hybrid",
            enable_isolation=True,
            use_bm25=True,
            persist_directory="./data/test_hybrid_chroma",
            bm25_persist_directory="./data/test_hybrid_bm25"
        )
        
        # 清空
        hybrid.clear()
        assert hybrid.get_count() == 0
        print("✓ 创建和清空混合存储成功")
        
        # 添加文档
        docs = [
            Document(
                page_content="消防安全是城市管理的重要组成部分",
                metadata={"topic": "消防"}
            ),
            Document(
                page_content="民法典规定了公民的基本权利和义务",
                metadata={"topic": "民法"}
            ),
        ]
        embeddings = [[0.1] * 1024, [0.2] * 1024]
        
        ids = hybrid.add_documents(docs, embeddings)
        assert len(ids) == 2
        print(f"✓ 添加 {len(ids)} 个文档到混合存储")
        
        # 测试向量搜索
        vector_results = hybrid.vector_store.similarity_search([0.15] * 1024, k=2)
        print(f"✓ 向量搜索返回 {len(vector_results)} 个结果")
        
        # 测试 BM25 搜索
        if hybrid.bm25_index:
            bm25_results = hybrid.bm25_index.search("消防安全", k=2)
            print(f"✓ BM25 搜索返回 {len(bm25_results)} 个结果")
        
        # 清理
        hybrid.clear()
        print("\n✓ 混合存储测试通过")
        
    except Exception as e:
        print(f"❌ 混合存储测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_store_factory():
    """测试存储工厂"""
    print("\n" + "=" * 60)
    print("测试 6: 存储工厂")
    print("=" * 60)
    
    from agent_system.rag.stores.factory import create_vector_store, create_bm25_index, StoreProvider
    
    # 测试创建 ChromaDB
    try:
        chroma = create_vector_store(
            provider=StoreProvider.CHROMA,
            user_id="factory_test",
            collection_prefix="factory_test"
        )
        print(f"✓ 工厂创建 ChromaDB: {chroma.collection_name}")
        chroma.clear_collection()
    except Exception as e:
        print(f"⚠️ ChromaDB 创建失败: {e}")
    
    # 测试创建 BM25
    try:
        bm25 = create_bm25_index(
            user_id="factory_test",
            collection_prefix="factory_bm25"
        )
        print(f"✓ 工厂创建 BM25: {bm25.index_name}")
        bm25.clear()
    except Exception as e:
        print(f"⚠️ BM25 创建失败: {e}")
    
    # 测试自动选择
    try:
        auto_store = create_vector_store(
            provider=StoreProvider.AUTO,
            user_id="auto_test"
        )
        print(f"✓ 自动选择存储: {type(auto_store).__name__}")
        auto_store.clear_collection()
    except Exception as e:
        print(f"⚠️ 自动选择失败: {e}")
    
    print("\n✓ 存储工厂测试完成")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("       存储层测试套件")
    print("=" * 70)
    
    tests = [
        ("用户隔离逻辑", test_user_isolation),
        ("BM25 索引", test_bm25_index),
        ("ChromaDB 存储", test_chroma_store),
        ("Milvus 存储", test_milvus_store),
        ("混合存储", test_hybrid_store),
        ("存储工厂", test_store_factory),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ 测试失败: {name}")
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
