"""SQLAlchemy 2.0 ORM models — single source of truth for the schema.

See docs/hld.md §HLD 2 (column definitions) and docs/lld.md §LLD 5
(conventions: Text over String(N), string+CHECK over Postgres ENUM,
TIMESTAMPTZ with server_default=now(), CASCADE on FKs to filings/companies,
HNSW with pgvector defaults).
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 1536


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "companies"

    cik: Mapped[str] = mapped_column(Text, primary_key=True)  # CHAR(10) zero-padded
    ticker: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    sic: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    filings: Mapped[list[Filing]] = relationship(
        back_populates="company", cascade="all, delete-orphan", passive_deletes=True
    )
    xbrl_facts: Mapped[list[XbrlFact]] = relationship(
        back_populates="company", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (CheckConstraint("char_length(cik) = 10", name="ck_companies_cik_len"),)


class Filing(Base):
    __tablename__ = "filings"

    accession_no: Mapped[str] = mapped_column(Text, primary_key=True)
    cik: Mapped[str] = mapped_column(
        Text, ForeignKey("companies.cik", ondelete="CASCADE"), nullable=False
    )
    form_type: Mapped[str] = mapped_column(Text, nullable=False)
    filing_date: Mapped[date] = mapped_column(Date, nullable=False)
    period_of_report: Mapped[date] = mapped_column(Date, nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    primary_doc_url: Mapped[str] = mapped_column(Text, nullable=False)
    local_html_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    ingestion_status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    company: Mapped[Company] = relationship(back_populates="filings")
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="filing", cascade="all, delete-orphan", passive_deletes=True
    )
    xbrl_facts: Mapped[list[XbrlFact]] = relationship(
        back_populates="filing", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        CheckConstraint(
            "ingestion_status IN ('pending','parsed','embedded','failed')",
            name="ck_filings_ingestion_status",
        ),
        Index("ix_filings_cik_fiscal_year_desc", "cik", "fiscal_year"),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    accession_no: Mapped[str] = mapped_column(
        Text, ForeignKey("filings.accession_no", ondelete="CASCADE"), nullable=False
    )
    cik: Mapped[str] = mapped_column(Text, nullable=False)  # denormalized from companies
    ticker: Mapped[str | None] = mapped_column(Text, nullable=True)  # denormalized
    company_name: Mapped[str | None] = mapped_column(Text, nullable=True)  # denormalized
    fiscal_year: Mapped[int | None] = mapped_column(Integer, nullable=True)  # denormalized
    filing_date: Mapped[date | None] = mapped_column(Date, nullable=True)  # denormalized
    section_title: Mapped[str | None] = mapped_column(Text, nullable=True)  # extracted sub-heading
    item: Mapped[str] = mapped_column(Text, nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)

    filing: Mapped[Filing] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("accession_no", "item", "seq", name="uq_chunks_accession_item_seq"),
        CheckConstraint("item IN ('1A','7','8')", name="ck_chunks_item"),
        Index("ix_chunks_cik_item", "cik", "item"),
        Index("ix_chunks_cik_item_year", "cik", "item", "fiscal_year"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class XbrlFact(Base):
    __tablename__ = "xbrl_facts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    cik: Mapped[str] = mapped_column(
        Text, ForeignKey("companies.cik", ondelete="CASCADE"), nullable=False
    )
    concept: Mapped[str] = mapped_column(Text, nullable=False)
    unit: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[Decimal] = mapped_column(Numeric(28, 4), nullable=False)
    period_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    fiscal_period: Mapped[str] = mapped_column(Text, nullable=False)
    accession_no: Mapped[str] = mapped_column(
        Text, ForeignKey("filings.accession_no", ondelete="CASCADE"), nullable=False
    )
    form: Mapped[str] = mapped_column(Text, nullable=False)

    company: Mapped[Company] = relationship(back_populates="xbrl_facts")
    filing: Mapped[Filing] = relationship(back_populates="xbrl_facts")

    __table_args__ = (
        UniqueConstraint(
            "cik", "concept", "period_end", "accession_no", name="uq_xbrl_natural_key"
        ),
        CheckConstraint("fiscal_period IN ('FY','Q1','Q2','Q3')", name="ck_xbrl_fiscal_period"),
        Index(
            "ix_xbrl_lookup",
            "cik",
            "concept",
            "fiscal_period",
            "fiscal_year",
        ),
    )


class AnswerCache(Base):
    __tablename__ = "answer_cache"

    cache_key: Mapped[str] = mapped_column(Text, primary_key=True)
    final_answer: Mapped[str] = mapped_column(Text, nullable=False)
    claims_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class QueryLog(Base):
    __tablename__ = "query_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(Text, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    question_type: Mapped[str] = mapped_column(Text, nullable=False)
    cik: Mapped[str | None] = mapped_column(Text, nullable=True)
    fiscal_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    refused: Mapped[bool] = mapped_column(Boolean, nullable=False)
    refusal_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    langfuse_trace_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            "question_type IN ('A_numerical','B_qualitative','REFUSE')",
            name="ck_query_log_question_type",
        ),
        Index("ix_query_log_created_at_desc", "created_at"),
    )


class AnonUsage(Base):
    """Lifetime per-IP request counter for the anonymous /ask preview.

    One row per hashed IP (sha256(ip || ANON_HASH_SALT)). Raw IPs are
    never stored. See docs/anon_hardening_plan.md §5.
    """

    __tablename__ = "anon_usage"

    ip_hash: Mapped[str] = mapped_column(Text, primary_key=True)
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint("char_length(ip_hash) = 64", name="ck_anon_usage_hash_len"),
        CheckConstraint("count >= 0", name="ck_anon_usage_count_nonneg"),
    )
