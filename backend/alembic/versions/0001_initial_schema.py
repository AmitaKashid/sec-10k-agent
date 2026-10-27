"""initial schema: companies, filings, chunks, xbrl_facts, answer_cache, query_log

Revision ID: 0001
Revises:
Create Date: 2026-05-01

Hand-written first migration per LLD 5 — alembic autogenerate cannot
detect the pgvector extension requirement, so the CREATE EXTENSION
runs before any table that uses VECTOR(1536).
"""

from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 1536


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "companies",
        sa.Column("cik", sa.Text(), primary_key=True),
        sa.Column("ticker", sa.Text(), nullable=False, unique=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("sic", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint("char_length(cik) = 10", name="ck_companies_cik_len"),
    )

    op.create_table(
        "filings",
        sa.Column("accession_no", sa.Text(), primary_key=True),
        sa.Column(
            "cik",
            sa.Text(),
            sa.ForeignKey("companies.cik", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("form_type", sa.Text(), nullable=False),
        sa.Column("filing_date", sa.Date(), nullable=False),
        sa.Column("period_of_report", sa.Date(), nullable=False),
        sa.Column("fiscal_year", sa.Integer(), nullable=False),
        sa.Column("primary_doc_url", sa.Text(), nullable=False),
        sa.Column("local_html_path", sa.Text(), nullable=True),
        sa.Column("ingestion_status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "ingestion_status IN ('pending','parsed','embedded','failed')",
            name="ck_filings_ingestion_status",
        ),
    )
    op.execute("CREATE INDEX ix_filings_cik_fiscal_year_desc ON filings (cik, fiscal_year DESC)")

    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "accession_no",
            sa.Text(),
            sa.ForeignKey("filings.accession_no", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("cik", sa.Text(), nullable=False),
        sa.Column("item", sa.Text(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.UniqueConstraint("accession_no", "item", "seq", name="uq_chunks_accession_item_seq"),
        sa.CheckConstraint("item IN ('1A','7','8')", name="ck_chunks_item"),
    )
    op.create_index("ix_chunks_cik_item", "chunks", ["cik", "item"])
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )

    op.create_table(
        "xbrl_facts",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "cik",
            sa.Text(),
            sa.ForeignKey("companies.cik", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("concept", sa.Text(), nullable=False),
        sa.Column("unit", sa.Text(), nullable=False),
        sa.Column("value", sa.Numeric(28, 4), nullable=False),
        sa.Column("period_start", sa.Date(), nullable=True),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("fiscal_year", sa.Integer(), nullable=False),
        sa.Column("fiscal_period", sa.Text(), nullable=False),
        sa.Column(
            "accession_no",
            sa.Text(),
            sa.ForeignKey("filings.accession_no", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("form", sa.Text(), nullable=False),
        sa.UniqueConstraint(
            "cik", "concept", "period_end", "accession_no", name="uq_xbrl_natural_key"
        ),
        sa.CheckConstraint("fiscal_period IN ('FY','Q1','Q2','Q3')", name="ck_xbrl_fiscal_period"),
    )
    op.execute(
        "CREATE INDEX ix_xbrl_lookup ON xbrl_facts (cik, concept, fiscal_period, fiscal_year DESC)"
    )

    op.create_table(
        "answer_cache",
        sa.Column("cache_key", sa.Text(), primary_key=True),
        sa.Column("final_answer", sa.Text(), nullable=False),
        sa.Column("claims_json", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("hit_count", sa.Integer(), nullable=False, server_default="0"),
    )

    op.create_table(
        "query_log",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("request_id", sa.Text(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("question_type", sa.Text(), nullable=False),
        sa.Column("cik", sa.Text(), nullable=True),
        sa.Column("fiscal_year", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("total_cost_usd", sa.Numeric(10, 6), nullable=True),
        sa.Column("refused", sa.Boolean(), nullable=False),
        sa.Column("refusal_reason", sa.Text(), nullable=True),
        sa.Column("langfuse_trace_id", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "question_type IN ('A_numerical','B_qualitative','REFUSE')",
            name="ck_query_log_question_type",
        ),
    )
    op.execute("CREATE INDEX ix_query_log_created_at_desc ON query_log (created_at DESC)")


def downgrade() -> None:
    op.drop_table("query_log")
    op.drop_table("answer_cache")
    op.drop_table("xbrl_facts")
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks")
    op.drop_index("ix_chunks_cik_item", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("ix_filings_cik_fiscal_year_desc", table_name="filings")
    op.drop_table("filings")
    op.drop_table("companies")
    # pgvector extension intentionally not dropped — may be used by other schemas.
