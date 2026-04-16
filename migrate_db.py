#!/usr/bin/env python3
"""
Database migration script to add new columns to the Claim model.
Run this after updating the models.py file.
"""

import sqlite3
import os
from database.models import db
from app import app

def migrate_database():
    """Add new columns to the claim table."""
    db_path = os.path.join(app.instance_path, 'insurance.db')

    if not os.path.exists(db_path):
        print("Database not found. Creating new database...")
        with app.app_context():
            db.create_all()
        return

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if new columns already exist
        cursor.execute("PRAGMA table_info(claim)")
        columns = [column[1] for column in cursor.fetchall()]

        new_columns = [
            ('vehicle_type', 'TEXT'),
            ('damage_location', 'TEXT'),
            ('damage_severity', 'TEXT'),
            ('fraud_analysis', 'TEXT')
        ]

        for column_name, column_type in new_columns:
            if column_name not in columns:
                print(f"Adding column: {column_name}")
                cursor.execute(f"ALTER TABLE claim ADD COLUMN {column_name} {column_type}")
            else:
                print(f"Column {column_name} already exists")

        conn.commit()
        print("Database migration completed successfully!")

    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()

    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database()