"""Add anon_usage table for the lifetime per-IP /ask preview cap.

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-21

Single row per hashed IP. Counter increments atomically on each anon
/ask call; once it reaches ANON_LIFETIME_LIMIT the row acts as a
permanent block until manually reset. Raw IPs are never stored —
ip_hash = sha256(ip || ANON_HASH_SALT). See
docs/anon_hardening_plan.md §5.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "anon_usage",
        sa.Column("ip_hash", sa.Text(), primary_key=True),
        sa.Column("count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "first_seen",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "last_seen",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("char_length(ip_hash) = 64", name="ck_anon_usage_hash_len"),
        sa.CheckConstraint("count >= 0", name="ck_anon_usage_count_nonneg"),
    )


def downgrade() -> None:
    op.drop_table("anon_usage")
