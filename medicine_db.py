"""
medicine_db.py — SQLite database for drug interaction history
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medicine_db.sqlite")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            medications TEXT NOT NULL,
            severity TEXT NOT NULL,
            details TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS medicines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT
        )
    ''')
    c.execute("SELECT COUNT(*) FROM medicines")
    if c.fetchone()[0] == 0:
        samples = [
            ("Aspirin",      "Pain reliever and anti-inflammatory"),
            ("Ibuprofen",    "NSAID anti-inflammatory"),
            ("Paracetamol",  "Fever reducer and pain reliever"),
            ("Amoxicillin",  "Antibiotic for bacterial infections"),
            ("Warfarin",     "Anticoagulant blood thinner"),
            ("Metformin",    "Type 2 diabetes medication"),
            ("Atorvastatin", "Cholesterol-lowering statin"),
            ("Lisinopril",   "ACE inhibitor for hypertension"),
        ]
        c.executemany("INSERT INTO medicines (name, description) VALUES (?, ?)", samples)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("Medicine DB initialised.")
