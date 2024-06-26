"""empty message

Revision ID: b3d2a87ec36d
Revises: f624d0186ad2
Create Date: 2024-03-10 20:33:06.300510

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b3d2a87ec36d"
down_revision: Union[str, None] = "f624d0186ad2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("album", sa.Column("source_url", sa.String(), nullable=False))
    op.alter_column(
        "album",
        "added_at",
        existing_type=sa.DATE(),
        type_=sa.DateTime(),
        existing_nullable=False,
    )
    op.alter_column(
        "artist",
        "added_at",
        existing_type=sa.DATE(),
        type_=sa.DateTime(),
        existing_nullable=False,
    )
    op.alter_column(
        "song",
        "added_at",
        existing_type=sa.DATE(),
        type_=sa.DateTime(),
        existing_nullable=False,
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "song",
        "added_at",
        existing_type=sa.DateTime(),
        type_=sa.DATE(),
        existing_nullable=False,
    )
    op.alter_column(
        "artist",
        "added_at",
        existing_type=sa.DateTime(),
        type_=sa.DATE(),
        existing_nullable=False,
    )
    op.alter_column(
        "album",
        "added_at",
        existing_type=sa.DateTime(),
        type_=sa.DATE(),
        existing_nullable=False,
    )
    op.drop_column("album", "source_url")
    # ### end Alembic commands ###
