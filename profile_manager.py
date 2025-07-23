import sqlite3
import json
import os
from typing import Optional, Dict, Any

class ProfileManager:
    """
    Manages the persistence of user profiles in a SQLite database.
    This version is refactored to be thread-safe for use with web servers like Gradio
    by creating a new database connection for each operation.
    """
    def __init__(self, db_path: str):
        """
        Initializes the ProfileManager.

        Args:
            db_path: The path to the SQLite database file.
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._create_table()

    def _get_connection(self):
        """Returns a new database connection."""
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        """Creates the user_profiles table if it doesn't already exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL
                )
            """)

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a user's profile from the database.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT profile_json FROM user_profiles WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None

    def save_profile(self, user_id: str, profile_data: Dict[str, Any]):
        """
        Saves or updates a user's profile in the database.
        """
        profile_json = json.dumps(profile_data)
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO user_profiles (user_id, profile_json) VALUES (?, ?)",
                (user_id, profile_json)
            )
