from pydantic import BaseModel, constr, EmailStr


class OCRIDCardEnrollment(BaseModel):
    front_id_card_base64: constr()


class OCRIDCardLogon(BaseModel):
    front_id_card_base64: constr()


class CheckFaceEnrollment(BaseModel):
    front_id_card_base64: constr()
    front_face_base64: constr()


class CheckFaceLogon(BaseModel):
    front_face_base64: constr()
