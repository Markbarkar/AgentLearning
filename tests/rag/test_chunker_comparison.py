"""
分块器对比测试

对比三种分块策略在法律文档上的效果：
1. RecursiveChunker - 递归字符分块（通用）
2. LegalChunker - 法律文档结构分块
3. SemanticChunker - 语义相似度分块

使用 data/law-data 目录下的真实法律文件进行测试
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 直接导入分块器，避免加载整个 rag 模块
from agent_system.rag.chunkers.legal_chunker import LegalChunker
from agent_system.rag.chunkers.recursive_chunker import RecursiveChunker
from agent_system.rag.chunkers.semantic_chunker import SemanticChunker
from agent_system.rag.extractors.legal_metadata import LegalMetadataExtractor


def read_docx(file_path: Path) -> str:
    """读取 docx 文件内容"""
    import docx
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])


def display_chunk(chunk, index: int, total: int, show_full: bool = False):
    """显示单个分块的详细信息"""
    content = chunk.page_content
    metadata = chunk.metadata
    
    print(f"\n┌{'─' * 78}┐")
    print(f"│ 分块 {index + 1}/{total}  [长度: {len(content)} 字符]")
    print(f"├{'─' * 78}┤")
    
    # 显示元数据
    chunker_type = metadata.get('chunker', 'unknown')
    if chunker_type == 'legal':
        print(f"│ 章: {metadata.get('chapter', 'N/A')}")
        print(f"│ 条款: {metadata.get('article_num', 'N/A')}")
        print(f"│ 上下文: {metadata.get('parent_context', 'N/A')}")
    elif chunker_type == 'semantic':
        print(f"│ 断点类型: {metadata.get('breakpoint_type', 'N/A')}")
        print(f"│ 阈值: {metadata.get('breakpoint_threshold', 'N/A')}")
    elif chunker_type == 'recursive':
        print(f"│ 分块方式: 递归字符分割")
    
    print(f"├{'─' * 78}┤")
    
    # 显示内容
    if show_full:
        display_content = content
    else:
        # 限制显示长度
        if len(content) > 500:
            display_content = content[:500] + "\n... (已截断，共 {} 字符) ...".format(len(content))
        else:
            display_content = content
    
    for line in display_content.split("\n"):
        if line.strip():
            # 每行最多显示76个字符
            while len(line) > 76:
                print(f"│ {line[:76]}")
                line = line[76:]
            print(f"│ {line}")
    
    print(f"└{'─' * 78}┘")


def display_chunks_summary(chunks: List, chunker_name: str):
    """显示分块摘要"""
    total_chars = sum(len(c.page_content) for c in chunks)
    avg_chars = total_chars // len(chunks) if chunks else 0
    min_chars = min(len(c.page_content) for c in chunks) if chunks else 0
    max_chars = max(len(c.page_content) for c in chunks) if chunks else 0
    
    print(f"\n【{chunker_name} 分块统计】")
    print(f"  • 分块数量: {len(chunks)}")
    print(f"  • 总字符数: {total_chars}")
    print(f"  • 平均长度: {avg_chars} 字符")
    print(f"  • 最小长度: {min_chars} 字符")
    print(f"  • 最大长度: {max_chars} 字符")


def test_single_file_comparison(
    file_path: Path,
    text: str,
    metadata: Dict[str, Any],
    use_semantic: bool = True,
    show_all_chunks: bool = True
):
    """对单个文件进行三种分块器的对比测试"""
    
    print("\n" + "=" * 80)
    print(f"📄 文件: {file_path.name}")
    print(f"   原文长度: {len(text)} 字符")
    print("=" * 80)
    
    results = {}
    
    # 1. 递归分块器
    print("\n" + "─" * 80)
    print("【1. RecursiveChunker - 递归字符分块】")
    print("─" * 80)
    
    start_time = time.time()
    recursive_chunker = RecursiveChunker(chunk_size=800, chunk_overlap=100)
    recursive_chunks = recursive_chunker.chunk(text, metadata.copy())
    recursive_time = time.time() - start_time
    
    display_chunks_summary(recursive_chunks, "RecursiveChunker")
    print(f"  • 耗时: {recursive_time:.3f} 秒")
    results["recursive"] = recursive_chunks
    
    if show_all_chunks:
        print("\n【RecursiveChunker 分块内容】")
        for i, chunk in enumerate(recursive_chunks):
            display_chunk(chunk, i, len(recursive_chunks))
    
    # 2. 法律分块器
    print("\n" + "─" * 80)
    print("【2. LegalChunker - 法律文档结构分块】")
    print("─" * 80)
    
    start_time = time.time()
    legal_chunker = LegalChunker(max_chunk_size=1500, min_chunk_size=100)
    legal_chunks = legal_chunker.chunk(text, metadata.copy())
    legal_time = time.time() - start_time
    
    display_chunks_summary(legal_chunks, "LegalChunker")
    print(f"  • 耗时: {legal_time:.3f} 秒")
    
    # 统计条款识别情况
    with_article = sum(1 for c in legal_chunks if c.metadata.get("article_num"))
    with_chapter = sum(1 for c in legal_chunks if c.metadata.get("chapter"))
    print(f"  • 识别到条款号: {with_article}/{len(legal_chunks)}")
    print(f"  • 识别到章节: {with_chapter}/{len(legal_chunks)}")
    results["legal"] = legal_chunks
    
    if show_all_chunks:
        print("\n【LegalChunker 分块内容】")
        for i, chunk in enumerate(legal_chunks):
            display_chunk(chunk, i, len(legal_chunks))
    
    # 3. 语义分块器
    if use_semantic:
        print("\n" + "─" * 80)
        print("【3. SemanticChunker - 语义相似度分块】")
        print("─" * 80)
        
        try:
            start_time = time.time()
            semantic_chunker = SemanticChunker(
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90,
                min_chunk_size=100,
                max_chunk_size=1500
            )
            semantic_chunks = semantic_chunker.chunk(text, metadata.copy())
            semantic_time = time.time() - start_time
            
            display_chunks_summary(semantic_chunks, "SemanticChunker")
            print(f"  • 耗时: {semantic_time:.3f} 秒")
            results["semantic"] = semantic_chunks
            
            if show_all_chunks:
                print("\n【SemanticChunker 分块内容】")
                for i, chunk in enumerate(semantic_chunks):
                    display_chunk(chunk, i, len(semantic_chunks))
                    
        except Exception as e:
            print(f"  ⚠️ 语义分块失败: {e}")
            print("  （需要设置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY 环境变量）")
            results["semantic"] = None
    
    # 对比总结
    print("\n" + "=" * 80)
    print("【分块器对比总结】")
    print("=" * 80)
    
    print(f"\n{'分块器':<20} {'分块数':<10} {'平均长度':<12} {'耗时':<10}")
    print("-" * 52)
    
    for name, chunks in results.items():
        if chunks:
            avg_len = sum(len(c.page_content) for c in chunks) // len(chunks)
            time_str = f"{recursive_time:.3f}s" if name == "recursive" else \
                       f"{legal_time:.3f}s" if name == "legal" else \
                       f"{semantic_time:.3f}s" if "semantic_time" in dir() else "N/A"
            print(f"{name:<20} {len(chunks):<10} {avg_len:<12} {time_str:<10}")
    
    return results


def test_comparison():
    """对比测试主函数"""
    print("\n" + "=" * 80)
    print("       分块器对比测试（使用 law-data 真实文件）")
    print("=" * 80)
    
    # 定位 law-data 目录
    law_data_dir = project_root / "data" / "law-data"
    
    if not law_data_dir.exists():
        print(f"❌ 测试目录不存在: {law_data_dir}")
        return False
    
    # 初始化元数据提取器
    extractor = LegalMetadataExtractor()
    
    # 选择测试文件
    test_files = [
        "深圳经济特区消防条例_.docx",
        "海南经济特区水条例_.docx",
    ]
    
    # 找到可用的文件
    available_files = []
    for f in test_files:
        fp = law_data_dir / f
        if fp.exists():
            available_files.append(fp)
    
    # 如果指定文件不存在，使用目录中的第一个 docx 文件
    if not available_files:
        for fp in law_data_dir.glob("*.docx"):
            available_files.append(fp)
            break
    
    if not available_files:
        print("❌ 没有找到可用的法律文件")
        return False
    
    # 检查是否有 API key
    has_openai = os.getenv("OPENAI_API_KEY") is not None
    has_dashscope = os.getenv("DASHSCOPE_API_KEY") is not None
    use_semantic = has_openai or has_dashscope
    
    if not use_semantic:
        print("\n⚠️ 未设置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY")
        print("   将跳过语义分块测试")
    
    # 只测试第一个文件（避免时间过长）
    file_path = available_files[0]
    
    try:
        text = read_docx(file_path)
        if not text:
            print(f"❌ 无法读取文件: {file_path.name}")
            return False
        
        metadata = extractor.extract(str(file_path), text)
        
        # 运行对比测试
        test_single_file_comparison(
            file_path=file_path,
            text=text,
            metadata=metadata,
            use_semantic=use_semantic,
            show_all_chunks=True  # 显示所有分块内容
        )
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_comparison():
    """快速对比测试（不显示详细内容）"""
    print("\n" + "=" * 80)
    print("       快速分块器对比测试")
    print("=" * 80)
    
    law_data_dir = project_root / "data" / "law-data"
    
    if not law_data_dir.exists():
        print(f"❌ 测试目录不存在: {law_data_dir}")
        return False
    
    extractor = LegalMetadataExtractor()
    
    # 测试多个文件
    docx_files = list(law_data_dir.glob("*.docx"))[:5]
    
    if not docx_files:
        print("❌ 没有找到 docx 文件")
        return False
    
    print(f"\n测试 {len(docx_files)} 个文件...")
    print(f"\n{'文件名':<40} {'递归':<8} {'法律':<8} {'条款识别率':<12}")
    print("-" * 70)
    
    for fp in docx_files:
        try:
            text = read_docx(fp)
            if not text:
                continue
            
            metadata = extractor.extract(str(fp), text)
            
            # 递归分块
            recursive = RecursiveChunker(chunk_size=800).chunk(text, metadata.copy())
            
            # 法律分块
            legal = LegalChunker().chunk(text, metadata.copy())
            
            # 条款识别率
            with_article = sum(1 for c in legal if c.metadata.get("article_num"))
            article_rate = f"{with_article}/{len(legal)}"
            
            # 截断文件名
            fname = fp.name[:38] + ".." if len(fp.name) > 40 else fp.name
            
            print(f"{fname:<40} {len(recursive):<8} {len(legal):<8} {article_rate:<12}")
            
        except Exception as e:
            print(f"{fp.name[:40]:<40} 错误: {e}")
    
    print("\n✓ 快速对比完成")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分块器对比测试")
    parser.add_argument("--quick", action="store_true", help="快速模式（不显示详细内容）")
    parser.add_argument("--no-semantic", action="store_true", help="跳过语义分块测试")
    args = parser.parse_args()
    
    if args.quick:
        success = test_quick_comparison()
    else:
        success = test_comparison()
    
    sys.exit(0 if success else 1)
