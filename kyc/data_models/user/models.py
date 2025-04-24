from sqlalchemy import Column, Integer, String, ARRAY, Float
from kyc.data_models.db import Base

from . import hashing


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    id_card = Column(String(255), unique=True)
    full_name = Column(String(255))
    dob = Column(String(255))
    sex = Column(String(255))
    nationality = Column(String(255))
    place_of_origin = Column(String(255))
    place_of_residence = Column(String(255))
    date_of_expiry = Column(String(255))

    front_id_card_base64 = Column(String)
    back_id_card_base64 = Column(String)

    front_face_base64 = Column(String)
    left_face_base64 = Column(String)
    right_face_base64 = Column(String)

    def __init__(self,
                id_card, full_name, dob, sex, nationality, place_of_origin, place_of_residence, date_of_expiry,
                front_id_card_base64, back_id_card_base64,
                front_face_base64, left_face_base64, right_face_base64,
                *args, **kwargs):
        self.id_card = hashing.get_password_hash(id_card)
        self.full_name = full_name
        self.dob = dob
        self.sex = sex
        self.nationality = nationality
        self.place_of_origin = place_of_origin
        self.place_of_residence = place_of_residence
        self.date_of_expiry = date_of_expiry

        self.front_id_card_base64 = front_id_card_base64
        self.back_id_card_base64 = back_id_card_base64

        self.front_face_base64 = front_face_base64
        self.left_face_base64 = left_face_base64
        self.right_face_base64 = right_face_base64

    def check_idcard(self, id_card):
        return hashing.verify_password(self.id_card, id_card)
