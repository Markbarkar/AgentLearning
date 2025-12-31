"""
用户 MCP 配置服务

提供用户级 MCP 配置的 CRUD 操作和配置合并功能
复用 user_configs 表，config_key 格式为 mcp_server_{server_name}
"""

import json
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
from dataclasses import asdict

from ..config.mcp_config import load_mcp_config, MCPServerConfig


# config_key 前缀
MCP_CONFIG_KEY_PREFIX = "mcp_server_"


def _get_config_key(server_name: str) -> str:
    """生成配置键名"""
    return f"{MCP_CONFIG_KEY_PREFIX}{server_name}"


def _parse_server_name(config_key: str) -> Optional[str]:
    """从配置键名解析服务器名称"""
    if config_key.startswith(MCP_CONFIG_KEY_PREFIX):
        return config_key[len(MCP_CONFIG_KEY_PREFIX):]
    return None


def get_user_mcp_config(
    db: Session, 
    user_id: int, 
    server_name: str
) -> Optional[Dict[str, Any]]:
    """
    获取用户单个 MCP 服务的覆盖配置
    
    Args:
        db: 数据库会话
        user_id: 用户 ID
        server_name: 服务器名称
        
    Returns:
        覆盖配置字典，不存在则返回 None
    """
    config_key = _get_config_key(server_name)
    
    result = db.execute(
        text("""
            SELECT config_value 
            FROM user_configs 
            WHERE user_id = :user_id AND config_key = :config_key
        """),
        {"user_id": user_id, "config_key": config_key}
    ).fetchone()
    
    if result and result[0]:
        try:
            return json.loads(result[0])
        except json.JSONDecodeError:
            return None
    return None


def list_user_mcp_configs(db: Session, user_id: int) -> Dict[str, Dict[str, Any]]:
    """
    获取用户所有 MCP 服务的覆盖配置
    
    Args:
        db: 数据库会话
        user_id: 用户 ID
        
    Returns:
        {服务器名称: 覆盖配置} 字典
    """
    results = db.execute(
        text("""
            SELECT config_key, config_value 
            FROM user_configs 
            WHERE user_id = :user_id AND config_key LIKE :prefix
        """),
        {"user_id": user_id, "prefix": f"{MCP_CONFIG_KEY_PREFIX}%"}
    ).fetchall()
    
    configs = {}
    for row in results:
        config_key, config_value = row
        server_name = _parse_server_name(config_key)
        if server_name and config_value:
            try:
                configs[server_name] = json.loads(config_value)
            except json.JSONDecodeError:
                continue
    
    return configs


def save_user_mcp_config(
    db: Session,
    user_id: int,
    server_name: str,
    config: Dict[str, Any]
) -> bool:
    """
    保存用户 MCP 服务的覆盖配置（插入或更新）
    
    Args:
        db: 数据库会话
        user_id: 用户 ID
        server_name: 服务器名称
        config: 覆盖配置（如 {"env": {"API_KEY": "xxx"}, "enabled": true}）
        
    Returns:
        是否保存成功
    """
    config_key = _get_config_key(server_name)
    config_value = json.dumps(config, ensure_ascii=False)
    
    try:
        # 检查是否已存在
        existing = db.execute(
            text("""
                SELECT id FROM user_configs 
                WHERE user_id = :user_id AND config_key = :config_key
            """),
            {"user_id": user_id, "config_key": config_key}
        ).fetchone()
        
        if existing:
            # 更新
            db.execute(
                text("""
                    UPDATE user_configs 
                    SET config_value = :config_value, update_time = NOW()
                    WHERE user_id = :user_id AND config_key = :config_key
                """),
                {"user_id": user_id, "config_key": config_key, "config_value": config_value}
            )
        else:
            # 插入
            db.execute(
                text("""
                    INSERT INTO user_configs (user_id, config_key, config_value, create_time, update_time)
                    VALUES (:user_id, :config_key, :config_value, NOW(), NOW())
                """),
                {"user_id": user_id, "config_key": config_key, "config_value": config_value}
            )
        
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"❌ 保存用户 MCP 配置失败: {e}")
        return False


def delete_user_mcp_config(db: Session, user_id: int, server_name: str) -> bool:
    """
    删除用户 MCP 服务的覆盖配置（恢复使用全局配置）
    
    Args:
        db: 数据库会话
        user_id: 用户 ID
        server_name: 服务器名称
        
    Returns:
        是否删除成功
    """
    config_key = _get_config_key(server_name)
    
    try:
        result = db.execute(
            text("""
                DELETE FROM user_configs 
                WHERE user_id = :user_id AND config_key = :config_key
            """),
            {"user_id": user_id, "config_key": config_key}
        )
        db.commit()
        return result.rowcount > 0
    except Exception as e:
        db.rollback()
        print(f"❌ 删除用户 MCP 配置失败: {e}")
        return False


def _deep_merge(base: dict, override: dict) -> dict:
    """
    深度合并两个字典
    
    override 中的值会覆盖 base 中的值
    对于嵌套字典会递归合并
    
    Args:
        base: 基础字典
        override: 覆盖字典
        
    Returns:
        合并后的字典
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并嵌套字典
            result[key] = _deep_merge(result[key], value)
        else:
            # 直接覆盖
            result[key] = value
    
    return result


def get_merged_mcp_config(
    db: Session, 
    user_id: int
) -> Dict[str, MCPServerConfig]:
    """
    获取合并后的 MCP 配置（全局模板 + 用户覆盖）
    
    合并逻辑：
    1. 加载全局模板（从 mcp_servers.json）
    2. 加载用户覆盖配置（从数据库）
    3. 对每个服务器进行深度合并
    
    Args:
        db: 数据库会话
        user_id: 用户 ID
        
    Returns:
        {服务器名称: MCPServerConfig} 合并后的配置
    """
    # 1. 加载全局模板
    global_configs = load_mcp_config()
    
    # 2. 加载用户覆盖配置
    user_overrides = list_user_mcp_configs(db, user_id)
    
    # 3. 合并配置
    merged = {}
    for server_name, global_config in global_configs.items():
        # 转换为字典进行合并
        base_dict = {
            "name": global_config.name,
            "command": global_config.command,
            "args": global_config.args,
            "env": global_config.env,
            "enabled": global_config.enabled,
            "description": global_config.description,
        }
        
        # 如果有用户覆盖，进行合并
        if server_name in user_overrides:
            merged_dict = _deep_merge(base_dict, user_overrides[server_name])
        else:
            merged_dict = base_dict
        
        # 转换回 MCPServerConfig
        merged[server_name] = MCPServerConfig(
            name=merged_dict.get("name", server_name),
            command=merged_dict.get("command", "npx"),
            args=merged_dict.get("args", []),
            env=merged_dict.get("env", {}),
            enabled=merged_dict.get("enabled", False),
            description=merged_dict.get("description", ""),
        )
    
    return merged


def get_merged_single_config(
    db: Session,
    user_id: int,
    server_name: str
) -> Optional[Dict[str, Any]]:
    """
    获取单个服务器的合并配置详情
    
    返回包含全局配置、用户覆盖和合并结果的完整信息
    
    Args:
        db: 数据库会话
        user_id: 用户 ID
        server_name: 服务器名称
        
    Returns:
        {
            "server_name": str,
            "global_config": dict,
            "user_override": dict,
            "merged_config": dict
        }
    """
    # 加载全局配置
    global_configs = load_mcp_config()
    
    if server_name not in global_configs:
        return None
    
    global_config = global_configs[server_name]
    global_dict = {
        "name": global_config.name,
        "command": global_config.command,
        "args": global_config.args,
        "env": global_config.env,
        "enabled": global_config.enabled,
        "description": global_config.description,
    }
    
    # 加载用户覆盖
    user_override = get_user_mcp_config(db, user_id, server_name) or {}
    
    # 合并
    merged_dict = _deep_merge(global_dict, user_override)
    
    return {
        "server_name": server_name,
        "global_config": global_dict,
        "user_override": user_override,
        "merged_config": merged_dict,
    }

