from pydantic import BaseModel, constr, EmailStr

class User(BaseModel):
    id_card: constr(min_length=12, max_length=12)
    full_name: constr(min_length=2, max_length=50)
    dob: constr()
    sex: constr()
    nationality: constr()
    place_of_origin: constr()
    place_of_residence: constr()
    date_of_expiry: constr()

    front_id_card_base64: constr()
    back_id_card_base64: constr()

    front_face_base64: constr()
    left_face_base64: constr()
    right_face_base64: constr()
