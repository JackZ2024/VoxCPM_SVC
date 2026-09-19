from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session

from sqlalchemy import DateTime, Integer, String, Text, JSON
from sqlalchemy.orm import (
    Mapped, 
    mapped_column, 
    relationship,
)

from api_config import(
    BASE_DIR,
    EXPIRED_SECONDS,
)

sqlite_db_path = BASE_DIR / 'api.db'
SQLALCHEMY_DATABASE_URL = f"sqlite:///{sqlite_db_path.as_posix()}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase) :
    pass


def get_db() :
    with SessionLocal() as db :
        yield db


class Task(Base) :
    __tablename__ = 'tasks'

    session_id : Mapped[str] = mapped_column(String, primary_key=True, index=True)
    request_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    result_files: Mapped[list[str]] = mapped_column(JSON, default=list())
    last_file: Mapped[str] = mapped_column(String(""), nullable=False)
    used_seed: Mapped[int] = mapped_column(Integer, nullable=False)

    def is_stopped(self) -> bool :
        return self.status in (
            'completed', 
            'failed', 
            'cancelled',
        )

    # def is_expired(self) :
    #     return (datetime.now(timezone.utc) - self.updated_at).seconds > EXPIRED_SECONDS

    def is_expired(self) -> bool:
        """
        检查任务是否过期。
        关键：确保参与比较的两个 datetime 对象时区信息一致。
        """
        
        # 1. 获取当前带时区的 UTC 时间 (Offset-Aware)
        now_aware = datetime.now(timezone.utc)
        
        # 2. 检查数据库读取到的 updated_at 是否有时区信息
        if self.updated_at.tzinfo is None:
            # 如果 SQLAlchemy 返回了 Naive 时间（通常是因为驱动未配置好），
            # 我们手动假设它是 UTC 并加上时区信息，使其变成 Aware
            self.updated_at = self.updated_at.replace(tzinfo=timezone.utc)
        else:
            # 如果已经有时区，确保将其转换为 UTC 以便比较（防止混合时区导致计算错误）
            # astimezone 不会改变时间点，只会改变表示方式
            self.updated_at = self.updated_at.astimezone(timezone.utc)

        # 3. 现在两者都是 Offset-Aware (带时区)，可以安全相减
        diff = now_aware - self.updated_at
        
        # 4. 使用 total_seconds() 比 .seconds 更准确（.seconds 只返回天数之后的秒数，如1天2小时会返回7200而不是93600）
        return diff.total_seconds() > EXPIRED_SECONDS
