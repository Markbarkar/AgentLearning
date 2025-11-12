"""
LangGraph 高级 Agent 示例

展示 LangGraph 的高级特性：
1. 复杂的控制流（条件分支、循环）
2. 人机交互（Human-in-the-loop）
3. 状态持久化（检查点）
4. 并行执行
5. 子图（Sub-graphs）
"""

import dotenv
dotenv.load_dotenv()

import operator
from typing import Annotated, TypedDict, Sequence, Literal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver


# ==================== 1. 定义状态 ====================

class AdvancedAgentState(TypedDict):
    """高级 Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_step: str  # 当前步骤
    task_type: str  # 任务类型（查询/购买/其他）
    confidence: float  # 置信度
    retry_count: int  # 重试次数
    user_approval: bool  # 用户是否批准


# ==================== 2. 定义节点 ====================

# 初始化 LLM
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    base_url="https://api.deepseek.com"
)


def analyze_task(state: AdvancedAgentState) -> dict:
    """
    任务分析节点
    
    分析用户任务，判断任务类型
    """
    messages = state["messages"]
    
    # 让 LLM 分析任务类型
    analysis_prompt = """
    分析以下任务，判断任务类型：
    - "query": 查询信息
    - "purchase": 购买操作
    - "other": 其他
    
    只返回类型，不要其他内容。
    
    任务: {task}
    """.format(task=messages[0].content)
    
    response = llm.invoke([HumanMessage(content=analysis_prompt)])
    task_type = response.content.strip().lower()
    
    print(f"📊 任务分析: {task_type}")
    
    return {
        "current_step": "analyzed",
        "task_type": task_type,
        "messages": [AIMessage(content=f"任务类型: {task_type}")]
    }


def handle_query(state: AdvancedAgentState) -> dict:
    """
    处理查询任务
    """
    print("🔍 执行查询操作...")
    
    # 模拟查询
    result = "查询结果：找到3趟火车票"
    
    return {
        "current_step": "query_completed",
        "confidence": 0.9,
        "messages": [AIMessage(content=result)]
    }


def handle_purchase(state: AdvancedAgentState) -> dict:
    """
    处理购买任务（需要人类批准）
    """
    print("💳 准备执行购买操作...")
    print("⚠️  购买操作需要人类批准！")
    
    return {
        "current_step": "awaiting_approval",
        "messages": [AIMessage(content="购买操作需要您的批准")]
    }


def wait_for_approval(state: AdvancedAgentState) -> dict:
    """
    等待人类批准
    
    在实际应用中，这里会暂停执行，等待用户输入
    """
    print("\n" + "=" * 70)
    print("⏸️  等待用户批准...")
    print("=" * 70)
    
    # 模拟用户输入
    user_input = input("是否批准购买？(yes/no): ").strip().lower()
    approved = user_input == "yes"
    
    if approved:
        print("✅ 用户已批准")
    else:
        print("❌ 用户拒绝")
    
    return {
        "user_approval": approved,
        "current_step": "approval_received"
    }


def execute_purchase(state: AdvancedAgentState) -> dict:
    """
    执行购买
    """
    print("💰 执行购买...")
    
    # 模拟购买
    result = "购买成功！订单号: 123456"
    
    return {
        "current_step": "purchase_completed",
        "messages": [AIMessage(content=result)]
    }


def handle_rejection(state: AdvancedAgentState) -> dict:
    """
    处理拒绝
    """
    print("🚫 购买已取消")
    
    return {
        "current_step": "cancelled",
        "messages": [AIMessage(content="购买已取消")]
    }


def handle_error(state: AdvancedAgentState) -> dict:
    """
    错误处理节点
    """
    retry_count = state.get("retry_count", 0)
    
    if retry_count < 3:
        print(f"⚠️  发生错误，重试 {retry_count + 1}/3")
        return {
            "retry_count": retry_count + 1,
            "current_step": "retrying"
        }
    else:
        print("❌ 达到最大重试次数")
        return {
            "current_step": "failed",
            "messages": [AIMessage(content="操作失败")]
        }


def summarize(state: AdvancedAgentState) -> dict:
    """
    总结节点
    """
    messages = state["messages"]
    
    print("\n" + "=" * 70)
    print("📝 生成总结...")
    print("=" * 70)
    
    summary_prompt = f"""
    根据以下对话历史，生成简洁的总结：
    
    {chr(10).join([f"- {msg.content}" for msg in messages if msg.content])}
    """
    
    response = llm.invoke([HumanMessage(content=summary_prompt)])
    
    return {
        "current_step": "completed",
        "messages": [AIMessage(content=f"总结: {response.content}")]
    }


# ==================== 3. 定义路由函数 ====================

def route_by_task_type(state: AdvancedAgentState) -> Literal["query", "purchase", "other"]:
    """
    根据任务类型路由
    """
    task_type = state.get("task_type", "other")
    
    if "query" in task_type or "查询" in task_type:
        return "query"
    elif "purchase" in task_type or "购买" in task_type:
        return "purchase"
    else:
        return "other"


def route_after_approval(state: AdvancedAgentState) -> Literal["approved", "rejected"]:
    """
    根据批准结果路由
    """
    if state.get("user_approval", False):
        return "approved"
    else:
        return "rejected"


def should_retry(state: AdvancedAgentState) -> Literal["retry", "give_up"]:
    """
    判断是否重试
    """
    retry_count = state.get("retry_count", 0)
    if retry_count < 3:
        return "retry"
    else:
        return "give_up"


# ==================== 4. 构建复杂状态图 ====================

def create_advanced_agent():
    """
    创建高级 Agent
    
    图结构：
                    START
                      ↓
                  analyze_task
                      ↓
              [根据任务类型路由]
              /       |        \
           query   purchase   other
             ↓         ↓        ↓
          完成   wait_approval  完成
                      ↓
              [根据批准结果]
                /          \
          approved      rejected
              ↓              ↓
        execute_purchase   取消
              ↓              ↓
                  summarize
                      ↓
                     END
    """
    # 创建状态图
    workflow = StateGraph(AdvancedAgentState)
    
    # 添加所有节点
    workflow.add_node("analyze_task", analyze_task)
    workflow.add_node("handle_query", handle_query)
    workflow.add_node("handle_purchase", handle_purchase)
    workflow.add_node("wait_for_approval", wait_for_approval)
    workflow.add_node("execute_purchase", execute_purchase)
    workflow.add_node("handle_rejection", handle_rejection)
    workflow.add_node("summarize", summarize)
    
    # 设置入口
    workflow.set_entry_point("analyze_task")
    
    # 添加条件边：根据任务类型路由
    workflow.add_conditional_edges(
        "analyze_task",
        route_by_task_type,
        {
            "query": "handle_query",
            "purchase": "handle_purchase",
            "other": "summarize"
        }
    )
    
    # 查询完成后 → 总结
    workflow.add_edge("handle_query", "summarize")
    
    # 购买流程 → 等待批准
    workflow.add_edge("handle_purchase", "wait_for_approval")
    
    # 根据批准结果路由
    workflow.add_conditional_edges(
        "wait_for_approval",
        route_after_approval,
        {
            "approved": "execute_purchase",
            "rejected": "handle_rejection"
        }
    )
    
    # 购买完成 → 总结
    workflow.add_edge("execute_purchase", "summarize")
    workflow.add_edge("handle_rejection", "summarize")
    
    # 总结完成 → 结束
    workflow.add_edge("summarize", END)
    
    # 编译（带检查点支持）
    memory = MemorySaver()  # 内存检查点
    return workflow.compile(checkpointer=memory)


# ==================== 5. 可视化图结构 ====================

def visualize_graph():
    """
    可视化 Agent 图结构
    
    需要安装: pip install pygraphviz
    """
    try:
        app = create_advanced_agent()
        
        # 生成 Mermaid 图
        print("\n" + "=" * 70)
        print("📊 Agent 图结构（Mermaid 格式）")
        print("=" * 70)
        print(app.get_graph().draw_mermaid())
        
    except Exception as e:
        print(f"无法生成可视化: {e}")


# ==================== 6. 运行 Agent ====================

def run_advanced_agent(task: str, thread_id: str = "default"):
    """
    运行高级 Agent
    
    参数:
        task: 任务描述
        thread_id: 线程 ID（用于检查点）
    """
    print("=" * 70)
    print("🚀 LangGraph 高级 Agent")
    print("=" * 70)
    print(f"\n📋 任务: {task}\n")
    
    # 创建 Agent
    app = create_advanced_agent()
    
    # 初始状态
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "current_step": "start",
        "task_type": "",
        "confidence": 0.0,
        "retry_count": 0,
        "user_approval": False
    }
    
    # 配置（用于检查点）
    config = {"configurable": {"thread_id": thread_id}}
    
    # 流式执行
    print("🔄 开始执行...\n")
    
    for output in app.stream(initial_state, config):
        for node_name, node_output in output.items():
            print(f"\n📍 当前节点: {node_name}")
            print("-" * 70)
    
    # 获取最终状态
    final_state = app.get_state(config)
    
    print("\n" + "=" * 70)
    print("✅ 执行完成")
    print("=" * 70)
    print(f"最终步骤: {final_state.values.get('current_step')}")
    
    # 打印最终消息
    if "messages" in final_state.values:
        last_message = final_state.values["messages"][-1]
        print(f"\n最终结果: {last_message.content}")


# ==================== 7. 主函数 ====================

def main():
    """主函数"""
    import sys
    
    # 可视化图结构
    if len(sys.argv) > 1 and sys.argv[1] == "--visualize":
        visualize_graph()
        return
    
    # 运行示例
    print("选择任务类型：")
    print("1. 查询火车票")
    print("2. 购买火车票")
    
    choice = input("\n请选择 (1/2): ").strip()
    
    if choice == "1":
        task = "帮我查询24年6月1日早上去上海的火车票"
    elif choice == "2":
        task = "帮我购买24年6月1日早上去上海的G123次列车的二等座"
    else:
        task = "你好"
    
    run_advanced_agent(task)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  程序被用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()




