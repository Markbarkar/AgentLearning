# 网页调试：
# 创建项目：adk create sample-agent【项目名】 --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY【apikey，可以不设环境变量】
# 启动项目：adk web --url_prefix {url_prefix}【地址，ex:http://localhost:8000】

import os
import dotenv
dotenv.load_dotenv()

try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    print(f"✅ Gemini API key setup complete. {GOOGLE_API_KEY}")
except Exception as e:
    print(
        f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

print("✅ ADK components imported successfully.")

# 配置重试请求配置
retry_config=types.HttpRetryOptions(
    attempts=5,  # 最大重试次数
    exp_base=7,  # 指数退避延迟的基数
    initial_delay=1, # 第一次重试前的初始延迟（秒）
    http_status_codes=[429, 500, 503, 504] # 重试的 HTTP 状态码
)

root_agent = Agent(
    name="helpful_assistant",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple agent that can answer general questions.",
    # agent的指导prompt，用于指导agent的行为
    instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
    tools=[google_search],
)

print("✅ Root Agent defined.")

# 创建一个内存中的runner，用于运行agent
runner = InMemoryRunner(agent=root_agent)

print("✅ Runner created.")

import asyncio

async def main():
    response = await runner.run_debug(
        "What is the weather in Foshan?"
    )
    print(response)

asyncio.run(main())
