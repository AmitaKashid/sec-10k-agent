"""Add ticker, company_name, fiscal_year metadata to chunks table.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-07

Denormalising these three fields onto chunks eliminates runtime joins,
enables fiscal-year-filtered vector search, and lets each ChunkHit carry
full citation context without querying filings/companies.

Columns are nullable so existing rows survive the migration; they are
populated on the next re-ingestion run (--force-reembed).
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("ticker", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("company_name", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("fiscal_year", sa.Integer(), nullable=True))

    # Index for year-scoped retrieval without a join to filings.
    op.create_index("ix_chunks_cik_item_year", "chunks", ["cik", "item", "fiscal_year"])


def downgrade() -> None:
    op.drop_index("ix_chunks_cik_item_year", table_name="chunks")
    op.drop_column("chunks", "fiscal_year")
    op.drop_column("chunks", "company_name")
    op.drop_column("chunks", "ticker")
