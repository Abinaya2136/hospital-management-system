import os
from datetime import datetime

import gradio as gr
import pandas as pd
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


CSV_FILE = "appointments.csv"
REQUIRED_COLUMNS = [
    "appointment_id", "patient_id", "doctor_id", "appointment_date",
    "appointment_time", "reason_for_visit", "status"
]
STATUS_OPTIONS = ["Scheduled", "Completed", "Cancelled", "No-show"]


def load_data():
    if os.path.exists(CSV_FILE):
        data = pd.read_csv(CSV_FILE)
        for col in REQUIRED_COLUMNS:
            if col not in data.columns:
                data[col] = ""
        return data[REQUIRED_COLUMNS].fillna("")
    return pd.DataFrame(columns=REQUIRED_COLUMNS)


appointments = load_data()


def save_data():
    appointments.to_csv(CSV_FILE, index=False)


def dashboard():
    total = len(appointments)
    completed = int((appointments["status"] == "Completed").sum()) if total else 0
    scheduled = int((appointments["status"] == "Scheduled").sum()) if total else 0
    cancelled = int((appointments["status"] == "Cancelled").sum()) if total else 0
    no_show = int((appointments["status"] == "No-show").sum()) if total else 0

    summary = (
        f"## Hospital Management Dashboard\n"
        f"**Total appointments:** {total}  \n"
        f"**Completed:** {completed}  |  **Scheduled:** {scheduled}  |  "
        f"**Cancelled:** {cancelled}  |  **No-show:** {no_show}"
    )
    if total:
        counts = appointments["status"].value_counts().reindex(STATUS_OPTIONS, fill_value=0)
        fig = px.pie(
            names=counts.index, values=counts.values,
            title="Appointment Status Distribution", hole=0.35
        )
    else:
        fig = px.pie(title="No appointment data available")
    return summary, fig, appointments.tail(10)


def view_appointments():
    return appointments


def add_appointment(aid, pid, did, adate, atime, reason, status):
    global appointments
    if not all([aid, pid, did, adate, atime, reason, status]):
        return "Please fill in all fields.", appointments
    if aid in appointments["appointment_id"].astype(str).values:
        return "Appointment ID already exists. Use a different ID.", appointments

    new_row = {
        "appointment_id": aid.strip(),
        "patient_id": pid.strip(),
        "doctor_id": did.strip(),
        "appointment_date": adate,
        "appointment_time": atime,
        "reason_for_visit": reason.strip(),
        "status": status
    }
    appointments = pd.concat([appointments, pd.DataFrame([new_row])], ignore_index=True)
    save_data()
    return "Appointment added successfully.", appointments


def search_appointments(query):
    if not query:
        return appointments
    q = str(query).lower()
    mask = appointments.astype(str).apply(
        lambda col: col.str.lower().str.contains(q, na=False)
    ).any(axis=1)
    return appointments[mask]


def update_status(aid, status):
    global appointments
    if not aid:
        return "Enter an appointment ID.", appointments
    mask = appointments["appointment_id"].astype(str) == str(aid).strip()
    if not mask.any():
        return "Appointment ID not found.", appointments
    appointments.loc[mask, "status"] = status
    save_data()
    return "Appointment status updated.", appointments


def delete_appointment(aid):
    global appointments
    if not aid:
        return "Enter an appointment ID.", appointments
    before = len(appointments)
    appointments = appointments[
        appointments["appointment_id"].astype(str) != str(aid).strip()
    ].reset_index(drop=True)
    if len(appointments) == before:
        return "Appointment ID not found.", appointments
    save_data()
    return "Appointment deleted.", appointments


def train_models():
    if len(appointments) < 20:
        return "At least 20 appointment records are recommended for model training.", None, None

    data = appointments.copy()
    dates = pd.to_datetime(data["appointment_date"], errors="coerce")
    times = pd.to_datetime(data["appointment_time"].astype(str), errors="coerce")
    data["month"] = dates.dt.month
    data["day_of_week"] = dates.dt.dayofweek
    data["hour"] = times.dt.hour

    features = ["doctor_id", "reason_for_visit", "month", "day_of_week", "hour"]
    data = data.dropna(subset=features + ["status"])
    if data["status"].nunique() < 2 or len(data) < 20:
        return "Not enough valid data or status categories to train the models.", None, None

    X = data[features]
    y = data["status"]
    counts = y.value_counts()
    if counts.min() < 2:
        return "Each status needs at least 2 records for a stratified train/test split.", None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    categorical = ["doctor_id", "reason_for_visit"]
    numeric = ["month", "day_of_week", "hour"]
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numeric)
    ])

    models = {
        "Existing 1 - Logistic Regression": LogisticRegression(max_iter=1000),
        "Existing 2 - Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Proposed - Random Forest": RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced"
        )
    }

    rows = []
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        rows.append({
            "Algorithm": name,
            "Accuracy": round(accuracy_score(y_test, pred), 3),
            "Precision (weighted)": round(precision_score(y_test, pred, average="weighted", zero_division=0), 3),
            "Recall (weighted)": round(recall_score(y_test, pred, average="weighted", zero_division=0), 3),
            "F1-score (weighted)": round(f1_score(y_test, pred, average="weighted", zero_division=0), 3)
        })

    results = pd.DataFrame(rows)
    chart = px.bar(
        results.melt(id_vars="Algorithm", var_name="Metric", value_name="Score"),
        x="Algorithm", y="Score", color="Metric", barmode="group",
        title="Comparison of Existing and Proposed Algorithms",
        range_y=[0, 1]
    )
    note = (
        "Model evaluation completed using a 75/25 stratified train/test split. "
        "Scores are based on this small sample dataset and are for academic demonstration; "
        "they are not clinical predictions."
    )
    return note, results, chart


with gr.Blocks(title="Hospital Management System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Hospital Management System")
    gr.Markdown("Appointment management dashboard with a machine-learning algorithm comparison.")

    with gr.Tab("Dashboard"):
        dashboard_text = gr.Markdown()
        dashboard_chart = gr.Plot()
        recent_table = gr.Dataframe(label="Recent Appointments", interactive=False)
        refresh = gr.Button("Refresh Dashboard")
        refresh.click(dashboard, outputs=[dashboard_text, dashboard_chart, recent_table])

    with gr.Tab("View and Search"):
        search_box = gr.Textbox(label="Search appointments", placeholder="Search by appointment ID, patient ID, doctor ID, reason, or status")
        search_btn = gr.Button("Search")
        all_table = gr.Dataframe(label="Appointments", interactive=False)
        search_btn.click(search_appointments, inputs=search_box, outputs=all_table)
        show_all = gr.Button("Show All Appointments")
        show_all.click(view_appointments, outputs=all_table)

    with gr.Tab("Add Appointment"):
        aid = gr.Textbox(label="Appointment ID", placeholder="Example: A201")
        pid = gr.Textbox(label="Patient ID", placeholder="Example: P061")
        did = gr.Textbox(label="Doctor ID", placeholder="Example: D009")
        adate = gr.Textbox(label="Appointment Date (YYYY-MM-DD)", placeholder="2026-11-21")
        atime = gr.Textbox(label="Appointment Time (HH:MM)", placeholder="10:30")
        reason = gr.Textbox(label="Reason for Visit")
        status = gr.Dropdown(STATUS_OPTIONS, label="Status", value="Scheduled")
        add_btn = gr.Button("Add Appointment", variant="primary")
        add_message = gr.Markdown()
        added_table = gr.Dataframe(label="Updated Appointments", interactive=False)
        add_btn.click(
            add_appointment,
            inputs=[aid, pid, did, adate, atime, reason, status],
            outputs=[add_message, added_table]
        )

    with gr.Tab("Update / Delete"):
        update_id = gr.Textbox(label="Appointment ID")
        update_status_value = gr.Dropdown(STATUS_OPTIONS, label="New Status", value="Completed")
        update_btn = gr.Button("Update Status")
        update_message = gr.Markdown()
        update_table = gr.Dataframe(label="Appointments", interactive=False)
        update_btn.click(update_status, inputs=[update_id, update_status_value], outputs=[update_message, update_table])

        gr.Markdown("### Delete Appointment")
        delete_id = gr.Textbox(label="Appointment ID to delete")
        delete_btn = gr.Button("Delete Appointment", variant="stop")
        delete_message = gr.Markdown()
        delete_table = gr.Dataframe(label="Updated Appointments", interactive=False)
        delete_btn.click(delete_appointment, inputs=delete_id, outputs=[delete_message, delete_table])

    with gr.Tab("Machine Learning"):
        gr.Markdown(
            "Compare two baseline algorithms with the proposed algorithm. "
            "All three classify appointment status using the available appointment fields."
        )
        train_btn = gr.Button("Train and Compare Algorithms", variant="primary")
        ml_message = gr.Markdown()
        ml_table = gr.Dataframe(label="Performance Comparison", interactive=False)
        ml_chart = gr.Plot()
        train_btn.click(train_models, outputs=[ml_message, ml_table, ml_chart])

    demo.load(dashboard, outputs=[dashboard_text, dashboard_chart, recent_table])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
