"""
MCP 服务器配置管理

从 mcp_servers.json 加载 MCP 服务器配置，支持：
- 环境变量替换 (${VAR_NAME})
- 跨平台命令处理 (Windows npx.cmd)
- 启用/禁用服务器
"""

import os
import re
import sys
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class MCPServerConfig:
    """MCP 服务器配置"""
    name: str
    command: str
    args: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""
    
    def get_command(self) -> str:
        """获取平台适配的命令"""
        # Windows 下 npx 需要使用 npx.cmd
        if sys.platform == "win32" and self.command == "npx":
            return "npx.cmd"
        return self.command
    
    def get_env(self) -> Dict[str, str]:
        """获取解析后的环境变量（合并系统环境变量）"""
        result = os.environ.copy()
        result.update(self.env)
        return result


def _expand_env_vars(value: str) -> str:
    """
    展开环境变量引用
    
    支持 ${VAR_NAME} 语法，从系统环境变量获取值
    如果环境变量不存在，返回空字符串
    """
    pattern = r'\$\{([^}]+)\}'
    
    def replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, "")
    
    return re.sub(pattern, replacer, value)


def _expand_env_in_dict(data: Dict[str, str]) -> Dict[str, str]:
    """展开字典中所有值的环境变量"""
    return {k: _expand_env_vars(v) for k, v in data.items()}


def _expand_env_in_list(data: List[str]) -> List[str]:
    """展开列表中所有值的环境变量"""
    return [_expand_env_vars(v) for v in data]


def load_mcp_config(config_path: Optional[str] = None) -> Dict[str, MCPServerConfig]:
    """
    加载 MCP 服务器配置
    
    Args:
        config_path: 配置文件路径，默认为 agent_system/config/mcp_servers.json
        
    Returns:
        服务器名称到配置对象的字典
    """
    if config_path is None:
        # 默认配置文件路径
        config_path = Path(__file__).parent / "mcp_servers.json"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"⚠️ MCP 配置文件不存在: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ MCP 配置文件解析失败: {e}")
        return {}
    
    servers = {}
    mcp_servers = data.get("mcpServers", {})
    
    for name, config in mcp_servers.items():
        # 展开环境变量
        args = _expand_env_in_list(config.get("args", []))
        env = _expand_env_in_dict(config.get("env", {}))
        
        server_config = MCPServerConfig(
            name=name,
            command=config.get("command", "npx"),
            args=args,
            env=env,
            enabled=config.get("enabled", True),
            description=config.get("description", "")
        )
        
        servers[name] = server_config
    
    return servers


def get_enabled_servers(config_path: Optional[str] = None) -> Dict[str, MCPServerConfig]:
    """
    获取所有启用的 MCP 服务器配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        仅包含 enabled=True 的服务器配置
    """
    all_servers = load_mcp_config(config_path)
    return {name: cfg for name, cfg in all_servers.items() if cfg.enabled}


def get_server_config(server_name: str, config_path: Optional[str] = None) -> Optional[MCPServerConfig]:
    """
    获取指定服务器的配置
    
    Args:
        server_name: 服务器名称
        config_path: 配置文件路径
        
    Returns:
        服务器配置，不存在则返回 None
    """
    all_servers = load_mcp_config(config_path)
    return all_servers.get(server_name)


# ==================== 配置文件写入操作 ====================

def _get_default_config_path() -> Path:
    """获取默认配置文件路径"""
    return Path(__file__).parent / "mcp_servers.json"


def _load_raw_config(config_path: Optional[str] = None) -> dict:
    """
    加载原始配置（不展开环境变量）
    
    用于配置的读写操作，保持原始格式
    """
    if config_path is None:
        config_path = _get_default_config_path()
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        return {"mcpServers": {}}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_mcp_config(data: dict, config_path: Optional[str] = None) -> bool:
    """
    保存配置到 JSON 文件
    
    Args:
        data: 完整的配置数据
        config_path: 配置文件路径
        
    Returns:
        是否保存成功
    """
    if config_path is None:
        config_path = _get_default_config_path()
    else:
        config_path = Path(config_path)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"❌ 保存配置失败: {e}")
        return False


def add_server_config(
    name: str,
    command: str,
    args: List[str],
    env: Optional[Dict[str, str]] = None,
    enabled: bool = True,
    description: str = "",
    config_path: Optional[str] = None
) -> bool:
    """
    添加新的 MCP 服务器配置
    
    Args:
        name: 服务器名称（唯一标识）
        command: 启动命令
        args: 命令参数
        env: 环境变量
        enabled: 是否启用
        description: 描述
        config_path: 配置文件路径
        
    Returns:
        是否添加成功
        
    Raises:
        ValueError: 服务器名称已存在
    """
    data = _load_raw_config(config_path)
    
    if name in data.get("mcpServers", {}):
        raise ValueError(f"服务器 '{name}' 已存在")
    
    if "mcpServers" not in data:
        data["mcpServers"] = {}
    
    server_config = {
        "command": command,
        "args": args,
        "enabled": enabled,
        "description": description
    }
    
    if env:
        server_config["env"] = env
    
    data["mcpServers"][name] = server_config
    
    return save_mcp_config(data, config_path)


def update_server_config(
    name: str,
    command: Optional[str] = None,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    enabled: Optional[bool] = None,
    description: Optional[str] = None,
    config_path: Optional[str] = None
) -> bool:
    """
    更新 MCP 服务器配置
    
    只更新提供的字段，其他字段保持不变
    
    Args:
        name: 服务器名称
        command: 启动命令
        args: 命令参数
        env: 环境变量
        enabled: 是否启用
        description: 描述
        config_path: 配置文件路径
        
    Returns:
        是否更新成功
        
    Raises:
        ValueError: 服务器不存在
    """
    data = _load_raw_config(config_path)
    
    if name not in data.get("mcpServers", {}):
        raise ValueError(f"服务器 '{name}' 不存在")
    
    server = data["mcpServers"][name]
    
    if command is not None:
        server["command"] = command
    if args is not None:
        server["args"] = args
    if env is not None:
        server["env"] = env
    if enabled is not None:
        server["enabled"] = enabled
    if description is not None:
        server["description"] = description
    
    return save_mcp_config(data, config_path)


def delete_server_config(name: str, config_path: Optional[str] = None) -> bool:
    """
    删除 MCP 服务器配置
    
    Args:
        name: 服务器名称
        config_path: 配置文件路径
        
    Returns:
        是否删除成功
        
    Raises:
        ValueError: 服务器不存在
    """
    data = _load_raw_config(config_path)
    
    if name not in data.get("mcpServers", {}):
        raise ValueError(f"服务器 '{name}' 不存在")
    
    del data["mcpServers"][name]
    
    return save_mcp_config(data, config_path)


def toggle_server(name: str, config_path: Optional[str] = None) -> bool:
    """
    切换 MCP 服务器的启用状态
    
    Args:
        name: 服务器名称
        config_path: 配置文件路径
        
    Returns:
        切换后的启用状态
        
    Raises:
        ValueError: 服务器不存在
    """
    data = _load_raw_config(config_path)
    
    if name not in data.get("mcpServers", {}):
        raise ValueError(f"服务器 '{name}' 不存在")
    
    server = data["mcpServers"][name]
    new_status = not server.get("enabled", True)
    server["enabled"] = new_status
    
    save_mcp_config(data, config_path)
    
    return new_status


def get_raw_server_config(name: str, config_path: Optional[str] = None) -> Optional[dict]:
    """
    获取服务器的原始配置（不展开环境变量）
    
    用于 API 返回，保持配置的原始格式
    
    Args:
        name: 服务器名称
        config_path: 配置文件路径
        
    Returns:
        原始配置字典，不存在则返回 None
    """
    data = _load_raw_config(config_path)
    servers = data.get("mcpServers", {})
    
    if name not in servers:
        return None
    
    config = servers[name].copy()
    config["name"] = name
    return config


def list_all_servers_raw(config_path: Optional[str] = None) -> List[dict]:
    """
    获取所有服务器的原始配置列表
    
    用于 API 返回，保持配置的原始格式
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        服务器配置列表
    """
    data = _load_raw_config(config_path)
    servers = data.get("mcpServers", {})
    
    result = []
    for name, config in servers.items():
        server_info = config.copy()
        server_info["name"] = name
        result.append(server_info)
    
    return result

