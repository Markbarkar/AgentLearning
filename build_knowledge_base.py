"""
知识库构建脚本

用于从文档目录批量构建知识库
"""

import argparse
import sys
from pathlib import Path

from agent_system.rag import KnowledgeBase
from agent_system.tools import Qwen25VLTools
from agent_system.config.settings import KNOWLEDGE_BASE_DOCS_DIR


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="构建法律文书知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        使用示例:
        # 从默认目录构建知识库
        python build_knowledge_base.py
        
        # 从指定目录构建知识库
        python build_knowledge_base.py --directory /path/to/docs
        
        # 清空已有数据后重新构建
        python build_knowledge_base.py --clear
        
        # 只处理 PDF 文件
        python build_knowledge_base.py --extensions .pdf
        
        # 不递归处理子目录
        python build_knowledge_base.py --no-recursive
                """
            )
    
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default=KNOWLEDGE_BASE_DOCS_DIR,
        help=f"文档目录路径（默认: {KNOWLEDGE_BASE_DOCS_DIR}）"
    )
    
    parser.add_argument(
        "-c", "--clear",
        action="store_true",
        help="清空已有数据后重新构建"
    )
    
    parser.add_argument(
        "-r", "--no-recursive",
        action="store_true",
        help="不递归处理子目录"
    )
    
    parser.add_argument(
        "-e", "--extensions",
        type=str,
        nargs="+",
        default=None,
        help="要处理的文件扩展名（如: .pdf .txt .docx）"
    )
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    directory = Path(args.directory)
    if not directory.exists():
        print(f"✗ 错误: 目录不存在: {directory}")
        print(f"\n提示: 请创建目录并放入文档，或使用 --directory 指定其他目录")
        sys.exit(1)
    
    print("=" * 70)
    print("法律文书知识库构建工具")
    print("=" * 70)
    print(f"\n配置信息:")
    print(f"  文档目录: {directory}")
    print(f"  递归处理: {'否' if args.no_recursive else '是'}")
    print(f"  清空已有数据: {'是' if args.clear else '否'}")
    if args.extensions:
        print(f"  文件类型: {', '.join(args.extensions)}")
    print()
    
    try:
        # 初始化 Qwen2.5-VL 工具（用于 PDF OCR）
        print("正在初始化 Qwen2.5-VL 工具...")
        vl_tools = Qwen25VLTools()
        print("✓ Qwen2.5-VL 工具初始化成功\n")
        
        # 初始化知识库
        print("正在初始化知识库...")
        kb = KnowledgeBase(vl_tools=vl_tools)
        print("✓ 知识库初始化成功\n")
        
        # 构建知识库
        result = kb.build_from_directory(
            directory=str(directory),
            recursive=not args.no_recursive,
            file_extensions=args.extensions,
            clear_existing=args.clear
        )
        
        # 显示结果
        if result["success"]:
            print("\n" + "=" * 70)
            print("✓ 知识库构建成功！")
            print("=" * 70)
            print(f"\n统计信息:")
            print(f"  新增文档块: {result['chunks_added']}")
            print(f"  总文档块数: {result['total_count']}")
            print("\n现在可以启动 API 服务使用 RAG 功能了！")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("✗ 知识库构建失败")
            print("=" * 70)
            print(f"错误信息: {result['message']}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n✗ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()



