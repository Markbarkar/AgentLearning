"""
向量化层测试

测试嵌入服务、缓存、批处理器和流水线
支持切换不同分块器进行效果对比
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_embedding_service():
    """测试嵌入服务"""
    print("\n" + "=" * 70)
    print("测试 1: EmbeddingService")
    print("=" * 70)
    
    from agent_system.rag.embedding.service import EmbeddingService, create_embedding_service
    
    # 检查可用的 API
    has_qwen = os.getenv("DASHSCOPE_API_KEY") is not None
    has_openai = os.getenv("OPENAI_API_KEY") is not None
    
    print(f"  Qwen API: {'✓ 可用' if has_qwen else '✗ 未配置'}")
    print(f"  OpenAI API: {'✓ 可用' if has_openai else '✗ 未配置'}")
    
    if not has_qwen and not has_openai:
        print("  ⚠️ 未配置任何 API Key，跳过嵌入服务测试")
        return True
    
    # 使用自动选择
    try:
        service = create_embedding_service(provider="auto")
        print(f"  使用的提供商: {service.provider.value}")
        print(f"  模型: {service.model}")
        print(f"  向量维度: {service.dimension}")
        
        # 测试单条嵌入
        test_text = "这是一条测试文本，用于验证嵌入服务是否正常工作。"
        
        start_time = time.time()
        embedding = service.embed_query(test_text)
        duration = time.time() - start_time
        
        print(f"\n  单条嵌入测试:")
        print(f"    文本长度: {len(test_text)} 字符")
        print(f"    向量维度: {len(embedding)}")
        print(f"    耗时: {duration:.3f} 秒")
        print(f"    向量前5维: {embedding[:5]}")
        
        # 测试批量嵌入
        batch_texts = [
            "第一条测试文本",
            "第二条测试文本",
            "第三条测试文本",
        ]
        
        start_time = time.time()
        embeddings = service.embed_documents(batch_texts)
        duration = time.time() - start_time
        
        print(f"\n  批量嵌入测试:")
        print(f"    文本数量: {len(batch_texts)}")
        print(f"    嵌入数量: {len(embeddings)}")
        print(f"    耗时: {duration:.3f} 秒")
        
        print("\n✓ 嵌入服务测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 嵌入服务测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_cache():
    """测试嵌入缓存"""
    print("\n" + "=" * 70)
    print("测试 2: EmbeddingCache")
    print("=" * 70)
    
    from agent_system.rag.embedding.cache import EmbeddingCache
    import tempfile
    
    # 创建临时缓存目录
    with tempfile.TemporaryDirectory() as cache_dir:
        cache = EmbeddingCache(
            cache_dir=cache_dir,
            max_memory_items=100
        )
        
        # 测试设置和获取
        test_text = "测试文本"
        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # 第一次获取（应该未命中）
        result = cache.get(test_text)
        assert result is None, "首次获取应该返回 None"
        print("  ✓ 首次获取返回 None")
        
        # 设置缓存
        cache.set(test_text, test_embedding)
        print("  ✓ 缓存设置成功")
        
        # 第二次获取（应该命中）
        result = cache.get(test_text)
        assert result == test_embedding, "应该返回缓存的向量"
        print("  ✓ 缓存命中成功")
        
        # 测试批量操作
        texts = ["文本1", "文本2", "文本3"]
        embeddings = [[0.1] * 5, [0.2] * 5, [0.3] * 5]
        
        cache.set_batch(texts, embeddings)
        hits, misses = cache.get_batch(texts + ["文本4"])
        
        assert len(hits) == 3, f"应该命中3个，实际 {len(hits)}"
        assert len(misses) == 1, f"应该未命中1个，实际 {len(misses)}"
        print(f"  ✓ 批量操作: 命中 {len(hits)}, 未命中 {len(misses)}")
        
        # 测试统计信息
        stats = cache.get_stats()
        print(f"  ✓ 缓存统计: hits={stats['hits']}, misses={stats['misses']}, hit_rate={stats['hit_rate']:.2%}")
        
        print("\n✓ 缓存测试通过")
        return True


def test_batch_processor():
    """测试批处理器"""
    print("\n" + "=" * 70)
    print("测试 3: BatchProcessor")
    print("=" * 70)
    
    has_api = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not has_api:
        print("  ⚠️ 未配置 API Key，跳过批处理器测试")
        return True
    
    from agent_system.rag.embedding.service import create_embedding_service
    from agent_system.rag.embedding.batcher import BatchProcessor
    from agent_system.rag.embedding.cache import EmbeddingCache
    from langchain_core.documents import Document
    import tempfile
    
    try:
        service = create_embedding_service(provider="auto")
        
        # 创建测试文档
        documents = [
            Document(page_content=f"这是第 {i+1} 个测试文档的内容。", metadata={"id": i})
            for i in range(5)
        ]
        
        # 创建批处理器
        processor = BatchProcessor(
            embedding_service=service,
            batch_size=10,
            max_workers=2,
            progress_callback=lambda done, total: print(f"    进度: {done}/{total}")
        )
        
        # 无缓存处理
        print("\n  无缓存处理:")
        result = processor.process(documents)
        
        print(f"    文档数: {len(result.documents)}")
        print(f"    嵌入数: {len(result.embeddings)}")
        print(f"    成功数: {result.success_count}")
        print(f"    错误数: {result.error_count}")
        print(f"    耗时: {result.duration:.3f} 秒")
        
        # 有缓存处理
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = EmbeddingCache(cache_dir=cache_dir)
            
            print("\n  有缓存处理（第一次）:")
            result1 = processor.process(documents, cache)
            print(f"    耗时: {result1.duration:.3f} 秒")
            
            print("\n  有缓存处理（第二次，应全部命中）:")
            result2 = processor.process(documents, cache)
            print(f"    耗时: {result2.duration:.3f} 秒")
            print(f"    缓存命中率: {cache.get_stats()['hit_rate']:.2%}")
        
        print("\n✓ 批处理器测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 批处理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_pipeline():
    """测试嵌入流水线"""
    print("\n" + "=" * 70)
    print("测试 4: EmbeddingPipeline")
    print("=" * 70)
    
    has_api = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not has_api:
        print("  ⚠️ 未配置 API Key，跳过流水线测试")
        return True
    
    from agent_system.rag.embedding.pipeline import EmbeddingPipeline, PipelineConfig
    import tempfile
    
    # 测试文本
    test_text = """
第一章 总则

第一条 为了加强消防工作，预防火灾和减少火灾危害，保护人身、财产安全，维护公共安全，根据《中华人民共和国消防法》及有关法律、行政法规的基本原则，结合深圳经济特区实际，制定本条例。

第二条 本条例适用于深圳经济特区。

第三条 消防工作贯彻预防为主、防消结合的方针，按照政府统一领导、部门依法监管、单位全面负责、公民积极参与的原则，实行消防安全责任制。

第二章 火灾预防

第四条 任何单位和个人都有维护消防安全、保护消防设施、预防火灾、报告火警的义务。
"""
    
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            # 创建流水线配置
            config = PipelineConfig(
                chunker_type="legal",
                embedding_provider="auto",
                enable_cache=True,
                cache_dir=cache_dir
            )
            
            # 创建流水线
            pipeline = EmbeddingPipeline(config=config)
            
            print("\n  【使用 LegalChunker】")
            result = pipeline.process_text(test_text, source="test_legal")
            
            print(f"    分块数: {len(result.documents)}")
            print(f"    嵌入数: {len(result.embeddings)}")
            print(f"    分块耗时: {result.stats.get('chunk_time', 0):.3f} 秒")
            print(f"    嵌入耗时: {result.stats.get('embed_time', 0):.3f} 秒")
            print(f"    总耗时: {result.stats.get('total_time', 0):.3f} 秒")
            
            # 显示分块详情
            print("\n    分块详情:")
            for i, doc in enumerate(result.documents[:3]):
                print(f"      [{i+1}] 长度={len(doc.page_content)}, 条款={doc.metadata.get('article_num', 'N/A')}")
            
            # 切换到 RecursiveChunker 对比
            print("\n  【切换到 RecursiveChunker】")
            pipeline.set_chunker("recursive", chunk_size=500)
            result2 = pipeline.process_text(test_text, source="test_recursive")
            
            print(f"    分块数: {len(result2.documents)}")
            print(f"    嵌入数: {len(result2.embeddings)}")
            print(f"    总耗时: {result2.stats.get('total_time', 0):.3f} 秒")
            
            # 使用 compare_chunkers
            print("\n  【分块器对比】")
            comparison = pipeline.compare_chunkers(
                test_text,
                chunker_types=["legal", "recursive"],
                metadata={"source": "comparison_test"}
            )
            
            for chunker_type, res in comparison.items():
                print(f"    {chunker_type}: {len(res.documents)} 块, 耗时 {res.stats.get('total_time', 0):.3f}s")
        
        print("\n✓ 流水线测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 流水线测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_with_real_file():
    """使用真实法律文件测试流水线"""
    print("\n" + "=" * 70)
    print("测试 5: 真实法律文件流水线处理")
    print("=" * 70)
    
    has_api = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not has_api:
        print("  ⚠️ 未配置 API Key，跳过测试")
        return True
    
    law_data_dir = project_root / "data" / "law-data"
    if not law_data_dir.exists():
        print(f"  ⚠️ 测试目录不存在: {law_data_dir}")
        return True
    
    # 找一个 docx 文件
    docx_files = list(law_data_dir.glob("*.docx"))[:1]
    if not docx_files:
        print("  ⚠️ 没有找到 docx 文件")
        return True
    
    from agent_system.rag.embedding.pipeline import EmbeddingPipeline, PipelineConfig
    import tempfile
    
    try:
        test_file = docx_files[0]
        print(f"\n  测试文件: {test_file.name}")
        
        with tempfile.TemporaryDirectory() as cache_dir:
            config = PipelineConfig(
                chunker_type="legal",
                embedding_provider="auto",
                enable_cache=True,
                cache_dir=cache_dir
            )
            
            pipeline = EmbeddingPipeline(config=config)
            
            # 处理文件
            print("\n  【LegalChunker 处理】")
            result = pipeline.process_file(test_file)
            
            print(f"    法规名: {result.metadata.get('law_name', 'N/A')}")
            print(f"    地区: {result.metadata.get('region', 'N/A')}")
            print(f"    类型: {result.metadata.get('law_type', 'N/A')}")
            print(f"    分块数: {len(result.documents)}")
            print(f"    总耗时: {result.stats.get('total_time', 0):.3f} 秒")
            
            # 显示部分分块
            print("\n    分块示例:")
            for i, doc in enumerate(result.documents[:3]):
                content_preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
                print(f"      [{i+1}] {doc.metadata.get('article_num', 'N/A')}: {content_preview}")
            
            # 对比 RecursiveChunker
            print("\n  【RecursiveChunker 处理】")
            pipeline.set_chunker("recursive", chunk_size=600)
            result2 = pipeline.process_file(test_file)
            
            print(f"    分块数: {len(result2.documents)}")
            print(f"    总耗时: {result2.stats.get('total_time', 0):.3f} 秒")
            
            # 对比分析
            print("\n  【对比分析】")
            print(f"    LegalChunker: {len(result.documents)} 块")
            print(f"    RecursiveChunker: {len(result2.documents)} 块")
            
            # 计算平均长度
            avg1 = sum(len(d.page_content) for d in result.documents) // len(result.documents)
            avg2 = sum(len(d.page_content) for d in result2.documents) // len(result2.documents)
            print(f"    LegalChunker 平均块长: {avg1} 字符")
            print(f"    RecursiveChunker 平均块长: {avg2} 字符")
        
        print("\n✓ 真实文件测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 真实文件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("       向量化层测试套件")
    print("=" * 70)
    
    tests = [
        ("嵌入服务测试", test_embedding_service),
        ("嵌入缓存测试", test_embedding_cache),
        ("批处理器测试", test_batch_processor),
        ("嵌入流水线测试", test_embedding_pipeline),
        ("真实文件测试", test_pipeline_with_real_file),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ {name} 异常: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"总测试结果: {passed} 通过, {failed} 失败")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
