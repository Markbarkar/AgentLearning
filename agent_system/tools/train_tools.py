"""
火车票相关工具

包含查询和购买火车票的工具实现
"""

from typing import List, Dict
from langchain_core.tools import StructuredTool


def search_train_ticket(
        origin: str,
        destination: str,
        date: str,
        departure_time_start: str,
        departure_time_end: str
) -> List[Dict[str, str]]:
    """
    查询火车票
    
    根据出发地、目的地、日期和时间范围查询可用的火车票
    
    参数:
        origin: 出发地
        destination: 目的地
        date: 出发日期 (格式: YYYY-MM-DD)
        departure_time_start: 出发时间范围开始 (格式: HH:MM)
        departure_time_end: 出发时间范围结束 (格式: HH:MM)
    
    返回:
        火车票列表，每个元素包含车次、时间、价格等信息
    
    注意:
        这里使用 mock 数据模拟数据库查询
        在实际应用中，应该调用真实的火车票查询 API
    """
    
    # 模拟数据库查询
    # 实际应用中应该调用 12306 API 或其他火车票查询服务
    mock_tickets = [
        {
            "train_number": "G1234",
            "origin": "北京",
            "destination": "上海",
            "departure_time": "2024-06-01 8:00",
            "arrival_time": "2024-06-01 12:00",
            "price": "100.00",
            "seat_type": "商务座",
        },
        {
            "train_number": "G5678",
            "origin": "北京",
            "destination": "上海",
            "departure_time": "2024-06-01 18:30",
            "arrival_time": "2024-06-01 22:30",
            "price": "100.00",
            "seat_type": "商务座",
        },
        {
            "train_number": "G9012",
            "origin": "北京",
            "destination": "上海",
            "departure_time": "2024-06-01 19:00",
            "arrival_time": "2024-06-01 23:00",
            "price": "100.00",
            "seat_type": "商务座",
        }
    ]
    
    return mock_tickets


def purchase_train_ticket(train_number: str) -> Dict:
    """
    购买火车票
    
    根据车次号购买火车票
    
    参数:
        train_number: 车次号 (如: G1234)
    
    返回:
        购买结果，包含状态、消息和座位信息
    
    注意:
        这里模拟购买操作
        实际应用中应该调用真实的购票 API
    """
    
    # 模拟购买操作
    # 实际应用中应该调用 12306 购票 API
    return {
        "result": "success",
        "message": "购买成功",
        "data": {
            "train_number": train_number,
            "seat_type": "商务座",
            "seat_number": "7-17A"
        }
    }


# 将 Python 函数包装成 LangChain 的 StructuredTool
# StructuredTool 会自动：
# 1. 从函数签名提取参数信息
# 2. 生成参数的 JSON Schema
# 3. 验证 Agent 传入的参数是否合法

search_train_ticket_tool = StructuredTool.from_function(
    func=search_train_ticket,
    name="查询火车票",
    description="查询指定日期可用的火车票。需要提供出发地、目的地、日期和时间范围。",
)

purchase_train_ticket_tool = StructuredTool.from_function(
    func=purchase_train_ticket,
    name="购买火车票",
    description="购买火车票。需要提供车次号。会返回购买结果(result)和座位号(seat_number)。",
)


