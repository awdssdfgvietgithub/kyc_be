from fastapi import FastAPI, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Query
from sqlalchemy.orm import Session

from kyc.data_models.user import router as user_router
from kyc.data_models.auth import router as auth_router
from kyc.data_models import db
from kyc.data_models.user import services, models
from kyc.data_models.user.models import User

app = FastAPI(
    title="HuitKYCAPP",
    description='This is UAT APIs docs',
    version='0.0.1',
)

templates = Jinja2Templates(directory="templates")

app.include_router(user_router.router)
app.include_router(auth_router.router)


@app.get("/web/users", response_class=HTMLResponse)
async def list_users(
    request: Request,
    page: int = Query(1, ge=1),
    name: str = Query(None),
    id_card: str = Query(None),
    database: Session = Depends(db.get_db)
):
    page_size = 10
    skip = (page - 1) * page_size

    users = await services.get_all_users(
        db=database,
        skip=skip,
        limit=page_size,
        name=name,
        id_card=id_card
    )

    total_query = database.query(User)
    if name:
        total_query = total_query.filter(User.full_name.ilike(f"%{name}%"))
    if id_card:
        total_query = total_query.filter(User.id_card.contains(id_card))

    total_users = total_query.count()
    total_pages = (total_users + page_size - 1) // page_size

    return templates.TemplateResponse("index.html", {
        "request": request,
        "users": users,
        "page": page,
        "total_pages": total_pages,
        "name": name,
        "id_card": id_card,
    })


@app.get("/web/user/{user_id}", response_class=HTMLResponse)
async def user_detail(request: Request, user_id: int, database: Session = Depends(db.get_db)):
    user = await services.get_user_by_id(user_id, database)
    return templates.TemplateResponse("user_detail.html", {"request": request, "user": user})