"""
数据库连接配置

提供 MySQL 数据库连接和会话管理
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 数据库连接配置
DB_HOST = os.getenv("DB_HOST", "192.168.2.106")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "lawyer_ai")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "zktm123")

# 构建数据库 URL
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

# 创建数据库引擎
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # 自动检测连接是否有效
    pool_recycle=3600,   # 每小时回收连接
    echo=False           # 生产环境关闭 SQL 日志
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 声明基类（如需定义 ORM 模型）
Base = declarative_base()


def get_db():
    """
    FastAPI 依赖注入：获取数据库会话
    
    使用方式：
        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_connection() -> bool:
    """
    测试数据库连接
    
    Returns:
        是否连接成功
    """
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False
