"""${message}
Revision ID: ${up_revision | default(<<'%s' % uuid.uuid4().hex)>}
Revises: ${down_revision | default(None)}
Create Date: ${create_date}
"""

from alembic import op
import sqlalchemy as sa

${imports if imports else ""}

def upgrade():
    ${upgrade if upgrade else "pass"}


def downgrade():
    ${downgrade if downgrade else "pass"}
