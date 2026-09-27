"""Prediction history persisted in SQLite via SQLAlchemy 2.0."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from fraudguard.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    source: Mapped[str] = mapped_column(
        String(32), index=True
    )  # api | api-batch | dashboard | dashboard-batch
    batch_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)

    step: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(16))
    amount: Mapped[float] = mapped_column(Float)
    oldbalanceOrg: Mapped[float] = mapped_column(Float)
    newbalanceOrig: Mapped[float] = mapped_column(Float)
    oldbalanceDest: Mapped[float] = mapped_column(Float)
    newbalanceDest: Mapped[float] = mapped_column(Float)

    fraud_probability: Mapped[float] = mapped_column(Float, index=True)
    is_fraud: Mapped[bool] = mapped_column(Boolean, index=True)
    threshold: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(16))
    reasons: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "source": self.source,
            "batch_id": self.batch_id,
            "step": self.step,
            "type": self.type,
            "amount": self.amount,
            "oldbalanceOrg": self.oldbalanceOrg,
            "newbalanceOrig": self.newbalanceOrig,
            "oldbalanceDest": self.oldbalanceDest,
            "newbalanceDest": self.newbalanceDest,
            "fraud_probability": self.fraud_probability,
            "is_fraud": self.is_fraud,
            "threshold": self.threshold,
            "risk_level": self.risk_level,
            "reasons": json.loads(self.reasons) if self.reasons else [],
            "model_version": self.model_version,
        }


class PredictionStore:
    """Thin repository around the ``predictions`` table."""

    def __init__(self, database_url: str = DATABASE_URL):
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, connect_args=connect_args, future=True)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    # ---- write ---------------------------------------------------------- #
    def save_frame(
        self,
        scored: pd.DataFrame,
        source: str,
        model_version: str | None = None,
        batch_id: str | None = None,
    ) -> str:
        """Persist rows from ``FraudPredictor.predict_frame`` output. Returns the batch id."""
        batch_id = batch_id or str(uuid.uuid4())
        rows = []
        for r in scored.to_dict("records"):
            rows.append(
                Prediction(
                    source=source,
                    batch_id=batch_id,
                    step=int(r["step"]),
                    type=str(r["type"]),
                    amount=float(r["amount"]),
                    oldbalanceOrg=float(r["oldbalanceOrg"]),
                    newbalanceOrig=float(r["newbalanceOrig"]),
                    oldbalanceDest=float(r["oldbalanceDest"]),
                    newbalanceDest=float(r["newbalanceDest"]),
                    fraud_probability=float(r["fraud_probability"]),
                    is_fraud=bool(r["is_fraud"]),
                    threshold=float(r.get("threshold", 0.5)),
                    risk_level=str(r.get("risk_level", "")),
                    reasons=json.dumps(list(r["reasons"]))
                    if "reasons" in r and r["reasons"] is not None
                    else None,
                    model_version=model_version,
                )
            )
        with self.Session() as s:
            s.add_all(rows)
            s.commit()
        return batch_id

    # ---- read ----------------------------------------------------------- #
    def recent(self, limit: int = 100, only_fraud: bool = False) -> list[dict]:
        stmt = (
            select(Prediction)
            .order_by(Prediction.created_at.desc(), Prediction.id.desc())
            .limit(limit)
        )
        if only_fraud:
            stmt = stmt.where(Prediction.is_fraud.is_(True))
        with self.Session() as s:
            return [p.to_dict() for p in s.scalars(stmt)]

    def recent_frame(self, limit: int = 100, only_fraud: bool = False) -> pd.DataFrame:
        rows = self.recent(limit=limit, only_fraud=only_fraud)
        return pd.DataFrame(rows)

    def stats(self) -> dict:
        with self.Session() as s:
            total = s.scalar(select(func.count(Prediction.id))) or 0
            flagged = (
                s.scalar(select(func.count(Prediction.id)).where(Prediction.is_fraud.is_(True)))
                or 0
            )
            batches = s.scalar(select(func.count(func.distinct(Prediction.batch_id)))) or 0
            avg_prob = s.scalar(select(func.avg(Prediction.fraud_probability)))
            last = s.scalar(select(func.max(Prediction.created_at)))
        return {
            "total_predictions": int(total),
            "flagged_fraud": int(flagged),
            "flag_rate": (flagged / total) if total else 0.0,
            "batches": int(batches),
            "avg_fraud_probability": float(avg_prob) if avg_prob is not None else None,
            "last_prediction_at": last.isoformat() if last else None,
        }

    def clear(self) -> int:
        with self.Session() as s:
            n = s.query(Prediction).delete()
            s.commit()
        return int(n)


def get_store(database_url: str = DATABASE_URL) -> PredictionStore:
    return PredictionStore(database_url)

