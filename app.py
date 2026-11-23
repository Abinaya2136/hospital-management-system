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
      

# UPDATE PATIENT
def update_patient(patient_id, name, age, gender, phone, address):
    data = get_patients()

    if str(patient_id) not in data["patient_id"].astype(str).values:
        return "Patient ID not found.", data

    idx = data.index[data["patient_id"].astype(str) == str(patient_id)][0]

    data.loc[idx, ["patient_name", "age", "gender", "phone", "address"]] = [
        name, age, gender, phone, address
    ]

    save_data(data, PF)
    return "Patient updated successfully!", get_patients()


# DELETE PATIENT
def delete_patient(patient_id):
    data = get_patients()

    if str(patient_id) not in data["patient_id"].astype(str).values:
        return "Patient ID not found.", data

    data = data[data["patient_id"].astype(str) != str(patient_id)]
    save_data(data, PF)

    return "Patient deleted successfully!", get_patients()

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
    data = get_doctors()
    if not str(name or "").strip():
        return "Please enter doctor name.", data
    did = make_id(data, "doctor_id", "D")
    row = [did, str(name).strip(), str(specialization or "").strip(),
           str(phone or "").strip(), availability or ""]
    data.loc[len(data)] = row
    save_data(data, DF)
    return f"Doctor added successfully. Doctor ID: {did}", get_doctors()


def find_doctors(query):
    data = get_doctors()
    query = str(query or "").strip()
    if query:
        data = data[data.astype(str).apply(
            lambda col: col.str.contains(query, case=False, na=False)
        ).any(axis=1)]
    return data


# -------------------- APPOINTMENT MODULE --------------------
def book_appointment(pid, did, date, time, reason):
    data = get_appointments()
    if not pid:
        return "Please select a patient.", data
    if not did:
        return "Please select a doctor.", data
    if not date or not time:
        return "Please enter appointment date and time.", data

    aid = make_id(data, "appointment_id", "A")
    data.loc[len(data)] = [
        aid, pid, did, str(date), str(time), reason or "Consultation", "Scheduled"
    ]
    save_data(data, AF)
    return f"Appointment booked successfully. Appointment ID: {aid}", get_appointments()


def find_appointments(query, status):
    data = get_appointments()
    query = str(query or "").strip()
    if query:
        data = data[data.astype(str).apply(
            lambda col: col.str.contains(query, case=False, na=False)
        ).any(axis=1)]
    if status != "All":
        data = data[data["status"] == status]
    return data


def change_appointment_status(aid, status):
    data = get_appointments()
    aid = str(aid or "").strip()
    if aid not in data["appointment_id"].values:
        return "Appointment ID not found.", data
    data.loc[data["appointment_id"] == aid, "status"] = status
    save_data(data, AF)
    return f"Appointment {aid} updated.", get_appointments()


# -------------------- DASHBOARD --------------------
def dashboard():
    patients = get_patients()
    doctors = get_doctors()
    appointments = get_appointments()

    counts = appointments["status"].value_counts().reindex(STATUSES, fill_value=0)
    summary = (
        "# Hospital Management Dashboard\n"
        f"**Total Patients:** {len(patients)}  |  "
        f"**Total Doctors:** {len(doctors)}  |  "
        f"**Total Appointments:** {len(appointments)}\n\n"
        + "  |  ".join([f"**{s}:** {counts[s]}" for s in STATUSES])
    )

    if len(appointments):
        chart = px.pie(
            names=counts.index, values=counts.values,
            title="Appointment Status Distribution", hole=0.35
        )
    else:
        chart = px.pie(
            names=["No appointments"], values=[1],
            title="Appointment Status Distribution"
        )
    return summary, chart, appointments.tail(10)


# -------------------- MACHINE LEARNING DEMO --------------------
def ml_demo():
    data = get_appointments()
    if len(data) < 10:
        return "Add at least 10 appointments to try the ML demo."

    # This is a simple demonstration using appointment details to predict status.
    features = data[["reason_for_visit", "doctor_id"]].fillna("")
    target = data["status"].fillna("Scheduled")

    if target.nunique() < 2:
        return "ML demo needs at least two different appointment statuses."

    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        transformer = ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"),
             ["reason_for_visit", "doctor_id"])
        ])
        model = Pipeline([
            ("transform", transformer),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        stratify = target if target.value_counts().min() >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=0.25, random_state=42, stratify=stratify
        )
        model.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(x_test))
        return f"Random Forest model trained successfully. Test accuracy: {accuracy:.2%}"
    except Exception as error:
        return f"ML demo could not run: {error}"


# -------------------- GRADIO USER INTERFACE --------------------
with gr.Blocks(title="Hospital Management System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Hospital Management System")
    gr.Markdown("Manage patients, doctors, appointments and view a simple dashboard.")

    with gr.Tabs():
        with gr.Tab("Dashboard"):
            dashboard_btn = gr.Button("Refresh Dashboard")
            dashboard_text = gr.Markdown()
            dashboard_chart = gr.Plot()
            recent_table = gr.Dataframe(label="Recent Appointments", interactive=False)
            dashboard_btn.click(
                dashboard,
                outputs=[dashboard_text, dashboard_chart, recent_table]
            )
            demo.load(
                dashboard,
                outputs=[dashboard_text, dashboard_chart, recent_table]
            )

        with gr.Tab("Patients"):
            gr.Markdown("## Add Patient")
            patient_name = gr.Textbox(label="Patient Name")
            patient_age = gr.Number(label="Age", precision=0)
            patient_gender = gr.Dropdown(
                ["Male", "Female", "Other"], label="Gender"
            )
            patient_phone = gr.Textbox(label="Phone")
            patient_address = gr.Textbox(label="Address")
            patient_add_btn = gr.Button("Add Patient", variant="primary")
            patient_message = gr.Textbox(label="Message", interactive=False)
            patient_table = gr.Dataframe(value=get_patients(), label="Patient Records")
            patient_add_btn.click(
                add_patient,
                inputs=[patient_name, patient_age, patient_gender,
                        patient_phone, patient_address],
                outputs=[patient_message, patient_table]
            )
              
            gr.Markdown("## Update Patient")
            update_id = gr.Textbox(label="Patient ID")
            update_name = gr.Textbox(label="Patient Name")
            update_age = gr.Number(label="Age", precision=0)
            update_gender = gr.Dropdown(
                ["Male", "Female", "Other"], label="Gender"
            )
            update_phone = gr.Textbox(label="Phone")
            update_address = gr.Textbox(label="Address")

            update_btn = gr.Button("Update Patient")
            update_btn.click(
                update_patient,
                inputs=[update_id, update_name, update_age,
                        update_gender, update_phone, update_address],
                outputs=[patient_message, patient_table]
            )

            gr.Markdown("## Delete Patient")
            delete_id = gr.Textbox(label="Patient ID")
            delete_btn = gr.Button("Delete Patient")
            delete_btn.click(
                delete_patient,
                inputs=delete_id,
                outputs=[patient_message, patient_table]
            )

            gr.Markdown("## Search Patients")
            patient_search = gr.Textbox(label="Search by ID, name, phone or address")
            patient_search_btn = gr.Button("Search Patients")
            patient_search_table = gr.Dataframe(label="Search Results")
            patient_search_btn.click(
                find_patients, inputs=patient_search, outputs=patient_search_table
            )

        with gr.Tab("Doctors"):
            gr.Markdown("## Add Doctor")
            doctor_name = gr.Textbox(label="Doctor Name")
            doctor_specialization = gr.Textbox(label="Specialization")
            doctor_phone = gr.Textbox(label="Phone")
            doctor_availability = gr.Dropdown(
                ["Morning", "Afternoon", "Evening", "Full Day"],
                label="Availability"
            )
            doctor_add_btn = gr.Button("Add Doctor", variant="primary")
            doctor_message = gr.Textbox(label="Message", interactive=False)
            doctor_table = gr.Dataframe(value=get_doctors(), label="Doctor Records")
            doctor_add_btn.click(
                add_doctor,
                inputs=[doctor_name, doctor_specialization,
                        doctor_phone, doctor_availability],
                outputs=[doctor_message, doctor_table]
            )

            gr.Markdown("## Search Doctors")
            doctor_search = gr.Textbox(label="Search by ID, name or specialization")
            doctor_search_btn = gr.Button("Search Doctors")
            doctor_search_table = gr.Dataframe(label="Search Results")
            doctor_search_btn.click(
                find_doctors, inputs=doctor_search, outputs=doctor_search_table
            )

        with gr.Tab("Appointments"):
            gr.Markdown("## Book Appointment")
            patient_dropdown = gr.Dropdown(
                choices=patient_choices(), label="Select Patient"
            )
            doctor_dropdown = gr.Dropdown(
                choices=doctor_choices(), label="Select Doctor"
            )
            appointment_date = gr.Textbox(
                label="Appointment Date", placeholder="YYYY-MM-DD"
            )
            appointment_time = gr.Textbox(
                label="Appointment Time", placeholder="e.g. 10:30 AM"
            )
            appointment_reason = gr.Dropdown(REASONS, value="Consultation",
                                              label="Reason for Visit")
            book_btn = gr.Button("Book Appointment", variant="primary")
            booking_message = gr.Textbox(label="Message", interactive=False)
            appointment_table = gr.Dataframe(
                value=get_appointments(), label="Appointment Records"
            )
            book_btn.click(
                book_appointment,
                inputs=[patient_dropdown, doctor_dropdown, appointment_date,
                        appointment_time, appointment_reason],
                outputs=[booking_message, appointment_table]
            )

            gr.Markdown("## Search Appointments")
            appointment_search = gr.Textbox(label="Search by appointment, patient or doctor ID")
            appointment_status_filter = gr.Dropdown(
                ["All"] + STATUSES, value="All", label="Filter by Status"
            )
            appointment_search_btn = gr.Button("Search Appointments")
            appointment_search_table = gr.Dataframe(label="Search Results")
            appointment_search_btn.click(
                find_appointments,
                inputs=[appointment_search, appointment_status_filter],
                outputs=appointment_search_table
            )

            gr.Markdown("## Update Appointment Status")
            update_id = gr.Textbox(label="Appointment ID")
            update_status = gr.Dropdown(STATUSES, label="New Status")
            update_btn = gr.Button("Update Status")
            update_message = gr.Textbox(label="Message", interactive=False)
            update_table = gr.Dataframe(label="Updated Appointments")
            update_btn.click(
                change_appointment_status,
                inputs=[update_id, update_status],
                outputs=[update_message, update_table]
            )

        with gr.Tab("Machine Learning"):
            gr.Markdown(
                "## Appointment Status Prediction Demo\n"
                "This demo trains a simple Random Forest model using the appointment records."
            )
            ml_btn = gr.Button("Train ML Model", variant="primary")
            ml_output = gr.Textbox(label="ML Result", lines=3, interactive=False)
            ml_btn.click(ml_demo, outputs=ml_output)

# -------------------- START SERVER --------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
