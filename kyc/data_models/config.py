import os

APP_ENV = os.getenv('APP_ENV', 'development')
DB_USERNAME = os.getenv('DATABASE_USERNAME', 'postgres')
DB_PASSWORD = os.getenv('DATABASE_PASSWORD', 'vietnguyen')
DB_HOST = os.getenv('DATABASE_HOST', 'localhost')
DB_NAME = os.getenv('DATABASE_NAME', 'huitkyc')
TEST_DB_NAME = os.getenv('DATABASE_NAME', 'huitkyc_test')
SQLALCHEMY_TRACK_MODIFICATIONS = False