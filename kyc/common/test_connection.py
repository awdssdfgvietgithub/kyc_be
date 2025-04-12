from sqlalchemy import create_engine
from kyc.data_models import config

DB_URL = f"postgresql://{config.DB_USERNAME}:{config.DB_PASSWORD}@{config.DB_HOST}/{config.DB_NAME}"

try:
    engine = create_engine(DB_URL)
    connection = engine.connect()
    print("✅ Kết nối thành công")
    connection.close()
except Exception as e:
    print(f"❌ Kết nối thất bại: {e}")