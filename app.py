import os
import pandas as pd
import gradio as gr
import plotly.express as px

# -------------------- FILES AND COLUMNS --------------------
PF, DF, AF = "patients.csv", "doctors.csv", "appointments.csv"

PC = ["patient_id", "patient_name", "age", "gender", "phone", "address"]
DC = ["doctor_id", "doctor_name", "specialization", "phone", "availability"]
AC = ["appointment_id", "patient_id", "doctor_id", "appointment_date",
      "appointment_time", "reason_for_visit", "status"]

STATUSES = ["Scheduled", "Completed", "Cancelled", "No-show"]
REASONS = ["Consultation", "Checkup", "Emergency", "Therapy", "Follow-up", "Other"]


# -------------------- DATA FUNCTIONS --------------------
def load_data(path, columns):
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)
    try:
        data = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        data = pd.DataFrame(columns=columns)

    for col in columns:
        if col not in data.columns:
            data[col] = ""
    return data[columns]


def save_data(data, path):
    data.to_csv(path, index=False)


def get_patients():
    return load_data(PF, PC)


def get_doctors():
    return load_data(DF, DC)


def get_appointments():
    return load_data(AF, AC)


def make_id(data, column, prefix):
    numbers = []
    for value in data[column].astype(str):
        if value.startswith(prefix):
            try:
                numbers.append(int(value[len(prefix):]))
            except ValueError:
                pass
    return f"{prefix}{max(numbers, default=0) + 1:03d}"


def patient_choices():
    data = get_patients()
    return [(f'{r["patient_id"]} - {r["patient_name"]}', r["patient_id"])
            for _, r in data.iterrows()]


def doctor_choices():
    data = get_doctors()
    return [(f'{r["doctor_id"]} - {r["doctor_name"]}', r["doctor_id"])
            for _, r in data.iterrows()]


# -------------------- PATIENT MODULE --------------------
def add_patient(name, age, gender, phone, address):
    data = get_patients()
    if not str(name or "").strip():
        return "Please enter patient name.", data
    pid = make_id(data, "patient_id", "P")
    row = [pid, str(name).strip(), str(age or ""), gender or "",
           str(phone or "").strip(), str(address or "").strip()]
    data.loc[len(data)] = row
    save_data(data, PF)
    return f"Patient added successfully. Patient ID: {pid}", get_patients()


def find_patients(query):
    data = get_patients()
    query = str(query or "").strip()
    if query:
        data = data[data.astype(str).apply(
            lambda col: col.str.contains(query, case=False, na=False)
        ).any(axis=1)]
    return data


# -------------------- DOCTOR MODULE --------------------
def add_doctor(name, specialization, phone, availability):
