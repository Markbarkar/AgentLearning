"""
MCP 服务器配置管理 API 路由

提供 MCP 服务器配置的增删改查和状态管理
"""

from fastapi import APIRouter, HTTPException
from typing import Optional

from ..schemas import MCPServerCreateRequest, MCPServerUpdateRequest
from ...config.mcp_config import (
    list_all_servers_raw,
    get_raw_server_config,
    add_server_config,
    update_server_config,
    delete_server_config,
    toggle_server,
    get_enabled_servers
)

# 创建路由器
router = APIRouter(prefix="/agent/mcp", tags=["MCP 服务器管理"])

# 全局变量：追踪当前连接的 MCP 适配器
# 由 dependencies.py 中的 create_qwen_vl_tools 更新
_connected_adapters = {}


def set_connected_adapters(adapters: dict):
    """设置当前连接的适配器（由外部调用）"""
    global _connected_adapters
    _connected_adapters = adapters


def get_connected_adapters() -> dict:
    """获取当前连接的适配器"""
    return _connected_adapters


# ==================== API 端点 ====================

@router.get("/servers")
async def list_servers():
    """
    获取所有 MCP 服务器配置
    
    返回所有配置的服务器列表，包括启用和禁用的
    """
    try:
        servers = list_all_servers_raw()
        return {
            "success": True,
            "total": len(servers),
            "servers": servers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/servers/{name}")
async def get_server(name: str):
    """
    获取单个 MCP 服务器配置详情
    
    Args:
        name: 服务器名称
    """
    try:
        config = get_raw_server_config(name)
        
        if config is None:
            raise HTTPException(status_code=404, detail=f"服务器 '{name}' 不存在")
        
        # 检查连接状态
        is_connected = name in _connected_adapters
        config["is_connected"] = is_connected
        
        # 如果已连接，添加工具信息
        if is_connected:
            adapter = _connected_adapters[name]
            config["tools"] = [
                {"name": t["name"], "description": t.get("description", "")}
                for t in adapter.tools_info
            ]
            config["tools_count"] = len(adapter.tools_info)
        
        return {
            "success": True,
            "server": config
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/servers")
async def create_server(request: MCPServerCreateRequest):
    """
    添加新的 MCP 服务器配置
    
    服务器名称必须唯一
    """
    try:
        success = add_server_config(
            name=request.name,
            command=request.command,
            args=request.args,
            env=request.env,
            enabled=request.enabled,
            description=request.description
        )
        
        if success:
            return {
                "success": True,
                "message": f"服务器 '{request.name}' 添加成功",
                "server": get_raw_server_config(request.name)
            }
        else:
            raise HTTPException(status_code=500, detail="保存配置失败")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/servers/{name}")
async def update_server(name: str, request: MCPServerUpdateRequest):
    """
    更新 MCP 服务器配置
    
    只更新提供的字段，其他字段保持不变
    
    Args:
        name: 服务器名称
    """
    try:
        success = update_server_config(
            name=name,
            command=request.command,
            args=request.args,
            env=request.env,
            enabled=request.enabled,
            description=request.description
        )
        
        if success:
            return {
                "success": True,
                "message": f"服务器 '{name}' 更新成功",
                "server": get_raw_server_config(name)
            }
        else:
            raise HTTPException(status_code=500, detail="保存配置失败")
            
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/servers/{name}")
async def delete_server(name: str):
    """
    删除 MCP 服务器配置
    
    Args:
        name: 服务器名称
    """
    try:
        # 检查是否正在连接中
        if name in _connected_adapters:
            raise HTTPException(
                status_code=400, 
                detail=f"服务器 '{name}' 正在使用中，请先停止服务"
            )
        
        success = delete_server_config(name)
        
        if success:
            return {
                "success": True,
                "message": f"服务器 '{name}' 已删除"
            }
        else:
            raise HTTPException(status_code=500, detail="删除配置失败")
            
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/servers/{name}/toggle")
async def toggle_server_status(name: str):
    """
    切换 MCP 服务器的启用/禁用状态
    
    Args:
        name: 服务器名称
    """
    try:
        new_status = toggle_server(name)
        
        return {
            "success": True,
            "message": f"服务器 '{name}' 已{'启用' if new_status else '禁用'}",
            "name": name,
            "enabled": new_status
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_connection_status():
    """
    获取当前 MCP 服务器连接状态
    
    返回已连接的服务器及其加载的工具信息
    """
    try:
        # 获取所有配置的服务器
        all_servers = list_all_servers_raw()
        
        # 获取已启用的服务器
        enabled_servers = get_enabled_servers()
        
        # 构建状态信息
        connected_servers = []
        disconnected_servers = []
        
        for server in all_servers:
            name = server["name"]
            server_status = {
                "name": name,
                "enabled": server.get("enabled", True),
                "description": server.get("description", "")
            }
            
            if name in _connected_adapters:
                adapter = _connected_adapters[name]
                server_status["is_connected"] = True
                server_status["tools_count"] = len(adapter.tools_info)
                server_status["tools"] = [t["name"] for t in adapter.tools_info]
                connected_servers.append(server_status)
            else:
                server_status["is_connected"] = False
                disconnected_servers.append(server_status)
        
        total_tools = sum(
            len(adapter.tools_info) 
            for adapter in _connected_adapters.values()
        )
        
        return {
            "success": True,
            # "summary": {
            #     "total_configured": len(all_servers),
            #     "total_enabled": len(enabled_servers),
            #     "total_connected": len(_connected_adapters),
            #     "total_tools": total_tools
            # },
            "connected": connected_servers,
            "disconnected": disconnected_servers
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

