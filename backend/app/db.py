import os
import psycopg


def get_dsn() -> str:
    return (
        f"host={os.getenv('DB_HOST', 'db')} "
        f"port={os.getenv('DB_PORT', '5432')} "
        f"user={os.getenv('DB_USER', 'postgres')} "
        f"password={os.getenv('DB_PASSWORD', 'postgres')} "
        f"dbname={os.getenv('DB_NAME', 'rh_assistant')}"
    )


def ping() -> bool:
    try:
        with psycopg.connect(get_dsn(), connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False
