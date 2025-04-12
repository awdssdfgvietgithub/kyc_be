from fastapi import HTTPException, status
from . import models
from . import schema


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
