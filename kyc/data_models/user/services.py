from . import models
from . import schema
from requests import Session


async def new_user_register(request, database) -> models.User:
    new_user = models.User(
        id_card=request.id_card,
        full_name=request.full_name,
        dob=request.dob,
        sex=request.sex,
        nationality=request.nationality,
        place_of_origin=request.place_of_origin,
        place_of_residence=request.place_of_residence,
        date_of_expiry=request.date_of_expiry,

        front_id_card_base64=request.front_id_card_base64,
        back_id_card_base64=request.back_id_card_base64,

        front_face_base64=request.front_face_base64,
        left_face_base64=request.left_face_base64,
        right_face_base64=request.right_face_base64,
    )
    database.add(new_user)
    database.commit()
    database.refresh(new_user)
    return new_user


async def get_all_users(db: Session, skip: int = 0, limit: int = 10, name: str = None, id_card: str = None):
    query = db.query(models.User)

    if name:
        query = query.filter(models.User.full_name.ilike(f"%{name}%"))
    if id_card:
        query = query.filter(models.User.id_card.contains(id_card))

    return query.offset(skip).limit(limit).all()


async def get_user_by_id(user_id: int, db: Session):
    return db.query(models.User).filter(models.User.id == user_id).first()
