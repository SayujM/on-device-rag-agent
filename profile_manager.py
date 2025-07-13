import sqlite3
import json
import os
from typing import Optional, Dict, Any

class ProfileManager:
    """
    Manages the persistence of user profiles in a SQLite database.
    This provides a simple, transactional key-value store for user data.
    """
    def __init__(self, db_path: str):
        """
        Initializes the ProfileManager and connects to the database.

        Args:
            db_path: The path to the SQLite database file.
        """
        # Ensure the directory for the database exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        """Creates the user_profiles table if it doesn't already exist."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL
                )
            """)

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a user's profile from the database.

        Args:
            user_id: The ID of the user to retrieve.

        Returns:
            A dictionary representing the user's profile, or None if not found.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT profile_json FROM user_profiles WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def save_profile(self, user_id: str, profile_data: Dict[str, Any]):
        """
        Saves or updates a user's profile in the database.

        Args:
            user_id: The ID of the user to save.
            profile_data: A dictionary representing the user's profile.
        """
        profile_json = json.dumps(profile_data)
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO user_profiles (user_id, profile_json) VALUES (?, ?)",
                (user_id, profile_json)
            )

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
