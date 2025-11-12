"""
Qwen2.5-VL Agent 系统入口文件

演示如何使用 Qwen2.5-VL 工具处理法律文书
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool

# 从各个模块导入需要的组件
from agent_system.config.settings import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_BASE_URL,
    LLM_SEED
)
from agent_system.tools import Qwen25VLTools, finish_tool
from agent_system.core import Agent


def create_qwen_vl_tools():
    """
    创建 Qwen2.5-VL 相关的 LangChain 工具
    
    Returns:
        工具列表
    """
    # 初始化 Qwen2.5-VL 工具集
    vl_tools = Qwen25VLTools()
    
    # 1. 提取关键信息工具
    extract_key_info_tool = Tool(
        name="extract_legal_key_info",
        description="""提取法律文书的关键信息。
        
输入参数：
- file_path: 文件路径（必填）

返回信息包括：
- title: 文件标题
- case_number: 案号
- case_reason: 案由
- court: 法院
- hearing_time: 开庭时间
- hearing_place: 开庭地点
- case_type: 案件类型
- lead_lawyer: 主办律师
- client_name: 委托人
- parties: 当事人

使用场景：快速了解法律文书的核心信息
示例输入：/Users/linzaizai/Desktop/Agent/doc/附件九.pdf""",
        func=lambda file_path: str(vl_tools.extract_key_info(file_path=file_path))
    )
    
    # 2. 提取文档全文工具
    extract_text_tool = Tool(
        name="extract_document_text",
        description="""提取文档的全部文字内容（OCR）。

输入参数：
- file_path: 文件路径（必填）

返回：文档每一页的文字内容列表

使用场景：需要获取文档完整文本内容时使用
示例输入：/Users/linzaizai/Desktop/Agent/doc/附件九.pdf""",
        func=lambda file_path: str(vl_tools.extract_text(file_path=file_path))
    )
    
    # 3. 标注法律文书工具（PDF）
    annotate_pdf_tool = Tool(
        name="annotate_legal_pdf",
        description="""标注 PDF 法律文书，提取指定字段及其在文档中的位置。

输入参数格式（用逗号分隔）：
file_path,key_fields

其中：
- file_path: 文件路径
- key_fields: 要提取的字段，用分号分隔，如 "案号;主办律师;法院"

支持的字段：
- 案号、案件编号
- 主办律师、承办律师
- 协办律师
- 文件类型、案件类型
- 当事人、原告、被告
- 委托人
- 法院
- 案由
- 开庭时间、开庭地点
- 落款日期、落款地点、落款人

返回：每个字段的内容和位置信息（边界框坐标）

使用场景：需要知道信息在文档中的具体位置时使用
示例输入：/Users/linzaizai/Desktop/Agent/doc/附件九.pdf,案号;法院;当事人""",
        func=lambda input_str: annotate_pdf_wrapper(vl_tools, input_str)
    )
    
    # 4. 识别表单工具
    recognize_form_tool = Tool(
        name="recognize_form",
        description="""识别法律案件表单/表格信息。

输入参数格式（用逗号分隔）：
file_path,table_type

其中：
- file_path: 文件路径
- table_type: 表单类型，可选值：
  * main: 案件主信息表
  * task: 任务与期限明细表
  * fee: 案件收费明细表
  * asset: 财产保全明细表
  * custom: 自定义表单（默认）

返回：结构化的表单数据

使用场景：识别结构化的表格信息
示例输入：/Users/linzaizai/Desktop/Agent/doc/附件九.pdf,custom""",
        func=lambda input_str: recognize_form_wrapper(vl_tools, input_str)
    )
    
    return [
        extract_key_info_tool,
        extract_text_tool,
        annotate_pdf_tool,
        recognize_form_tool,
        finish_tool
    ]


def annotate_pdf_wrapper(vl_tools, input_str):
    """
    标注 PDF 的包装函数
    
    Args:
        vl_tools: Qwen25VLTools 实例
        input_str: 输入字符串，格式为 "file_path,key_fields"
    
    Returns:
        标注结果的字符串表示
    """
    try:
        parts = input_str.split(',', 1)
        if len(parts) != 2:
            return "错误：输入格式应为 'file_path,key_fields'，例如：'/path/to/file.pdf,案号;法院;当事人'"
        
        file_path = parts[0].strip()
        key_fields = parts[1].strip().replace(';', ',')
        
        result = vl_tools.annotate_legal_pdf(
            file_path=file_path,
            key_fields=key_fields,
            temperature=0.1
        )
        
        return str(result)
    except Exception as e:
        return f"错误：{str(e)}"


def recognize_form_wrapper(vl_tools, input_str):
    """
    识别表单的包装函数
    
    Args:
        vl_tools: Qwen25VLTools 实例
        input_str: 输入字符串，格式为 "file_path,table_type"
    
    Returns:
        识别结果的字符串表示
    """
    try:
        parts = input_str.split(',')
        file_path = parts[0].strip()
        table_type = parts[1].strip() if len(parts) > 1 else "custom"
        
        result = vl_tools.recognize_form(
            file_path=file_path,
            table_type=table_type
        )
        
        return str(result)
    except Exception as e:
        return f"错误：{str(e)}"


def main():
    """
    主函数
    
    演示完整的 Qwen2.5-VL Agent 使用流程
    """
    
    print("=" * 60)
    print("🤖 Qwen2.5-VL Agent 系统启动")
    print("=" * 60)
    
    # 1. 初始化 LLM（大语言模型）
    print("\n📝 初始化 LLM...")
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        base_url=LLM_BASE_URL,
        model_kwargs={
            "seed": LLM_SEED
        }
    )
    print(f"✅ LLM 初始化完成: {LLM_MODEL}")
    
    # 2. 准备工具列表
    print("\n🔧 准备 Qwen2.5-VL 工具...")
    tools = create_qwen_vl_tools()
    print(f"✅ 工具准备完成，共 {len(tools)} 个工具")
    for tool in tools:
        print(f"   - {tool.name}")
    
    # 3. 创建 Agent 实例
    print("\n🤖 创建 Agent...")
    agent = Agent(llm=llm, tools=tools)
    print("✅ Agent 创建完成")
    
    # 4. 定义任务
    print("\n" + "=" * 60)
    print("📋 任务描述")
    print("=" * 60)
    
    # 任务：分析法律文书
    task = """请帮我分析 /Users/linzaizai/Desktop/Agent/doc/附件九.pdf 这份法律文书，
            我需要知道以下信息：
            1. 案号是什么？
            2. 涉及哪个法院？
            3. 开庭时间是什么时候？
            4. 主办律师是谁？
            5. 当事人有哪些？

            请使用合适的工具提取这些信息，并给我一个清晰的总结。"""
    
    print(f"任务: {task}")
    
    # 5. 运行 Agent
    print("\n" + "=" * 60)
    print("🚀 开始执行任务")
    print("=" * 60)
    print()
    
    result = agent.run(task)
    
    # 6. 输出最终结果
    print("\n" + "=" * 60)
    print("✨ 最终结果")
    print("=" * 60)
    print(result)
    print("\n" + "=" * 60)
    print("🎉 任务完成")
    print("=" * 60)


def test_simple_task():
    """
    测试简单任务
    """
    print("=" * 60)
    print("🧪 测试简单任务")
    print("=" * 60)
    
    # 初始化 LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        base_url=LLM_BASE_URL,
        model_kwargs={"seed": LLM_SEED}
    )
    
    # 准备工具
    tools = create_qwen_vl_tools()
    
    # 创建 Agent
    agent = Agent(llm=llm, tools=tools)
    
    # 简单任务：只提取关键信息
    task = "请提取 /Users/linzaizai/Desktop/Agent/doc/附件九.pdf 的关键信息，告诉我案号和法院。"
    
    print(f"\n任务: {task}\n")
    
    result = agent.run(task)
    
    print("\n结果:")
    print(result)


if __name__ == "__main__":
    # 运行主任务
    main()
    
    # 或者运行简单测试
    # test_simple_task()

