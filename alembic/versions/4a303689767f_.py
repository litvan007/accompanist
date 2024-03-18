"""empty message

Revision ID: 4a303689767f
Revises: 0017d75656ce
Create Date: 2024-03-19 00:14:13.688031

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4a303689767f'
down_revision: Union[str, None] = '0017d75656ce'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint('track_album_id_fkey', 'track', type_='foreignkey')
    op.create_foreign_key(None, 'track', 'album', ['album_id'], ['id'], ondelete='CASCADE')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'track', type_='foreignkey')
    op.create_foreign_key('track_album_id_fkey', 'track', 'album', ['album_id'], ['id'])
    # ### end Alembic commands ###
