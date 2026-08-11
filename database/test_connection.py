from sqlalchemy import text
from connection import get_engine

engine = get_engine()

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT current_database();"))
        print("Connected Successfully!")
        print("Database:", result.fetchone()[0])

except Exception as e:
    print("Connection Failed!")
    print(e)