"""Shared SQLAlchemy declarative base for every AI persistence adapter."""

from datetime import datetime

from sqlalchemy import DateTime, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .conventions import BIGSERIAL, _OnUpdateDateTime


class Base(DeclarativeBase):
    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        _OnUpdateDateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )


__all__ = ["Base"]
