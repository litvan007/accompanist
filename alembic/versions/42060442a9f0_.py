"""empty message

Revision ID: 42060442a9f0
Revises: b46fbe9d06e4
Create Date: 2024-03-13 11:02:07.772867

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '42060442a9f0'
down_revision: Union[str, None] = 'b46fbe9d06e4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('track',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('artist_id', sa.Integer(), nullable=False),
    sa.Column('album_id', sa.Integer(), nullable=False),
    sa.Column('added_at', sa.DateTime(), nullable=False),
    sa.Column('filename_vocals', sa.String(), nullable=False),
    sa.Column('filename_instrumental', sa.String(), nullable=False),
    sa.Column('number_in_album', sa.Integer(), nullable=False),
    sa.Column('duration', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['album_id'], ['album.id'], ),
    sa.ForeignKeyConstraint(['artist_id'], ['artist.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.drop_table('song')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('song',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('name', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('artist_id', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('album_id', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('added_at', postgresql.TIMESTAMP(), autoincrement=False, nullable=False),
    sa.Column('filename_original', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('filename_instrumental', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.ForeignKeyConstraint(['album_id'], ['album.id'], name='song_album_id_fkey'),
    sa.ForeignKeyConstraint(['artist_id'], ['artist.id'], name='song_artist_id_fkey'),
    sa.PrimaryKeyConstraint('id', name='song_pkey')
    )
    op.drop_table('track')
    # ### end Alembic commands ###