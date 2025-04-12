from fastapi import FastAPI
from kyc.data_models.user import router as user_router
from kyc.data_models.auth import router as auth_router

app = FastAPI(
    title="HuitKYCAPP",
    description='This is UAT APIs docs',
    version='0.0.1',
)

app.include_router(user_router.router)
app.include_router(auth_router.router)