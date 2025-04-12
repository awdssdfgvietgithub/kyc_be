from typing import Optional

from sqlalchemy.orm import Session

from .models import User


async def verify_id_card_exist(id_card, database: Session) -> Optional[User]:
    return database.query(User).filter(User.id_card == id_card).first()
