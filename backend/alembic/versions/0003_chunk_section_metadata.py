"""Add section_title and filing_date to chunks table.

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-07

section_title: the sub-heading within the 10-K item that the chunk falls
under (e.g. "Competition", "Cybersecurity Risks"). Extracted at ingestion
time by scanning for standalone short lines between paragraph breaks.

filing_date: the date the filing was submitted to SEC. Useful for
displaying precise citation context and for future time-scoped queries.

Both columns nullable — existing rows keep NULL until --force-reembed.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("section_title", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("filing_date", sa.Date(), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "filing_date")
    op.drop_column("chunks", "section_title")
