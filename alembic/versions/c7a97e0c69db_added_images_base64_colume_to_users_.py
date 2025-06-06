"""Added images base64 colume to users table

Revision ID: c7a97e0c69db
Revises: 3cbe4df997bc
Create Date: 2025-04-05 12:55:52.069263

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'c7a97e0c69db'
down_revision: Union[str, None] = '3cbe4df997bc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('users', sa.Column('front_id_card_base64', sa.String(), nullable=True))
    op.add_column('users', sa.Column('back_id_card_base64', sa.String(), nullable=True))
    op.add_column('users', sa.Column('front_face_base64', sa.String(), nullable=True))
    op.add_column('users', sa.Column('left_face_base64', sa.String(), nullable=True))
    op.add_column('users', sa.Column('right_face_base64', sa.String(), nullable=True))
    op.drop_column('users', 'embedding')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('users', sa.Column('embedding', postgresql.ARRAY(sa.DOUBLE_PRECISION(precision=53)), autoincrement=False, nullable=True))
    op.drop_column('users', 'right_face_base64')
    op.drop_column('users', 'left_face_base64')
    op.drop_column('users', 'front_face_base64')
    op.drop_column('users', 'back_id_card_base64')
    op.drop_column('users', 'front_id_card_base64')
    # ### end Alembic commands ###
