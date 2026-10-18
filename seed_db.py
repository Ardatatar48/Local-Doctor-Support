"""
Creates a random (SYNTHETIC / FICTIONAL) patient database.
Usage:  python seed_db.py          (adds 20 patients)
        python seed_db.py 50       (adds 50 patients)
Has no relation to real people; test data for homework only.
"""
import os
import random
import sqlite3
import sys
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hastalar.db")

MALE = ["James", "Michael", "David", "Robert", "John", "Daniel", "Thomas", "Mark",
        "Paul", "Kevin", "Brian", "Steven", "Andrew", "Peter", "George"]
FEMALE = ["Mary", "Emma", "Sophia", "Olivia", "Laura", "Anna", "Julia", "Sarah",
          "Emily", "Claire", "Hannah", "Grace", "Alice", "Nina", "Lucy"]
SURNAME = ["Smith", "Johnson", "Brown", "Taylor", "Miller", "Wilson", "Moore", "Clark",
           "Walker", "Young", "King", "Wright", "Scott", "Green", "Baker", "Adams"]

COMPLAINT = [
    "Radiating pain from the lower back down the leg for the last 3 months, worse in the mornings.",
    "Low back pain and numbness in the left leg after sitting for a long time.",
    "Severe low back pain when bending, tingling in the right leg.",
    "Low back pain that has not resolved for two weeks, difficulty bending forward.",
    "Low back and hip pain that started after lifting something heavy.",
    "Low back pain that wakes the patient at night, eases with walking.",
]
HISTORY = [
    ("HIV positive (under follow-up for 3 years), viral load suppressed.", "Dolutegravir + Lamivudine"),
    ("HIV positive, newly diagnosed, treatment not started yet.", "None"),
    ("Type 2 diabetes.", "Metformin 1000 mg twice daily"),
    ("Hypertension.", "Ramipril 5 mg once daily"),
    ("Type 2 diabetes and hypertension.", "Metformin, Amlodipine"),
    ("Osteoporosis.", "Calcium + vitamin D, Alendronate"),
    ("History of stomach ulcer.", "Pantoprazole 40 mg"),
    ("Asthma.", "Salbutamol inhaler"),
    ("Chronic kidney disease (stage 2).", "None"),
    ("No known chronic disease.", "None"),
    ("Hepatitis B carrier.", "Entecavir"),
    ("Obesity (BMI 33).", "None"),
]
ALLERGY = ["None", "Penicillin allergy", "Allergy to NSAIDs", "None", "None", "Sulfonamide allergy"]


def init_db(c):
    c.execute("""CREATE TABLE IF NOT EXISTS patients(
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
        age INTEGER, created TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS records(
        id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INTEGER,
        image_path TEXT, report_text TEXT, analysis TEXT, created TEXT)""")


def make_report(name, age, sex):
    history, meds = random.choice(HISTORY)
    return (
        f"Patient: {name}, {age} years old, {sex}.\n"
        f"Complaint: {random.choice(COMPLAINT)}\n"
        f"Medical history: {history}\n"
        f"Current medications: {meds}\n"
        f"Allergies: {random.choice(ALLERGY)}\n"
        f"Physical exam: Straight leg raise test {random.choice(['positive', 'negative'])}, "
        f"lumbar muscle spasm {random.choice(['present', 'absent'])}."
    )


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    with sqlite3.connect(DB_PATH) as c:
        init_db(c)
        for _ in range(n):
            sex = random.choice(["male", "female"])
            first = random.choice(MALE if sex == "male" else FEMALE)
            name = f"{first} {random.choice(SURNAME)}"
            age = random.randint(22, 80)
            t = (datetime.now() - timedelta(days=random.randint(0, 120))).isoformat(timespec="seconds")
            cur = c.execute("INSERT INTO patients(name, age, created) VALUES(?,?,?)", (name, age, t))
            c.execute(
                "INSERT INTO records(patient_id, image_path, report_text, analysis, created) VALUES(?,?,?,?,?)",
                (cur.lastrowid, "", make_report(name, age, sex), "", t),
            )
    print(f"{n} random patients added -> {DB_PATH}")


if __name__ == "__main__":
    main()
