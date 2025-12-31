"""
Qwen2.5-VL Agent API 服务

提供 HTTP API 接口，接收用户任务并返回 Agent 处理结果

主入口文件 - 负责组装和启动 FastAPI 应用
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 从 API 模块导入路由
from agent_system.api.routes import knowledge_base, documents, agent, mcp


# 创建 FastAPI 应用
app = FastAPI(
    title="Qwen2.5-VL Agent API",
    description="法律文书智能处理 Agent API",
    version="1.0.0"
)

# 配置CORS（如需启用，取消下面的注释）
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://192.168.2.106:6789",
#         "http://localhost:6789",
#         "http://www.zktmai.com:9093",
#         "http://www.zktmai.com:8001",
#         "http://www.zktmai.com:6789",
#         "http://120.237.13.172:9093",
#         "http://120.237.13.172:8001",
#         "http://120.237.13.172:6789",
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
#     expose_headers=["Content-Length", "Content-Range"],
#     max_age=3600,
# )

# 注册路由
app.include_router(knowledge_base.router)
app.include_router(documents.router)
app.include_router(agent.router)
app.include_router(mcp.router)


if __name__ == "__main__":
    import uvicorn
    from agent_system.config.settings import LLM_MODEL
    from agent_system.tools import Qwen25VLTools
    
    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )
