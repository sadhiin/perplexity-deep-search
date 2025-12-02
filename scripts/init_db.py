"""Initialize the project's local SQLite database."""
from backend.database.connection import init_db, DATABASE_URL

def main():
    print(f"Initializing database at {DATABASE_URL}...")
    init_db()
    print("Database initialized.")

if __name__ == "__main__":
    main()
