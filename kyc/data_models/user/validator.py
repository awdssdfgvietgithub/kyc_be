from typing import Optional, List, Type

from sqlalchemy.orm import Session

from .models import User


async def verify_id_card_exist(id_card, database: Session) -> Optional[User]:
    return database.query(User).filter(User.id_card == id_card).first()


async def get_all_users(database: Session) -> list[Type[User]]:
    return database.query(User).all()
