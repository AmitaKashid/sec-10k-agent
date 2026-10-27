"""Add GIN index on chunks.text for full-text keyword search.

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-07

Without this index, every keyword search leg of the hybrid retrieval
does a sequential scan recomputing to_tsvector per row. A GIN index
on the pre-computed tsvector lets Postgres satisfy @@ queries in
O(log n) and makes ts_rank lookups fast enough to meet the <5 s SLO.

Note: GIN indexes are write-slower than BTree but read-fast for
full-text. The chunk table is append-heavy at ingest time (not during
queries), so the write cost is acceptable.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS ix_chunks_text_gin "
            "ON chunks USING gin(to_tsvector('english', text))"
        )
    )


def downgrade() -> None:
    op.execute(sa.text("DROP INDEX IF EXISTS ix_chunks_text_gin"))
