"""
Action 数据模型

定义 Agent 执行的动作结构
使用 Pydantic 进行数据验证
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class Action(BaseModel):
    """
    Agent 执行的动作
    
    每一步 Agent 都会决定执行一个动作（调用一个工具）
    这个类定义了动作的数据结构
    
    属性:
        name: 要调用的工具名称
        args: 工具的参数（字典格式）
    
    示例:
        >>> action = Action(
        ...     name="查询火车票",
        ...     args={
        ...         "origin": "北京",
        ...         "destination": "上海",
        ...         "date": "2024-06-01"
        ...     }
        ... )
    """
    
    name: str = Field(
        description="工具或指令名称",
        examples=["查询火车票", "购买火车票", "FINISH"]
    )
    
    args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="工具或指令参数，由参数名称和参数值组成",
        examples=[
            {
                "origin": "北京",
                "destination": "上海",
                "date": "2024-06-01"
            }
        ]
    )
    
    class Config:
        """Pydantic 配置"""
        # JSON Schema 示例
        json_schema_extra = {
            "example": {
                "name": "查询火车票",
                "args": {
                    "origin": "北京",
                    "destination": "上海",
                    "date": "2024-06-01",
                    "departure_time_start": "08:00",
                    "departure_time_end": "12:00"
                }
            }
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"Action(name={self.name}, args={self.args})"
    
    def __repr__(self) -> str:
        """调试表示"""
        return self.__str__()


