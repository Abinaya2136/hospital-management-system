import os
import pandas as pd
import gradio as gr
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PF, DF, AF = 'patients.csv', 'doctors.csv', 'appointments.csv'
PC = ['patient_id','patient_name','age','gender','phone','address']
DC = ['doctor_id','doctor_name','specialization','phone','availability']
AC = ['appointment_id','patient_id','doctor_id','appointment_date','appointment_time','reason_for_visit','status']
STATUSES = ['Scheduled','Completed','Cancelled','No-show']
REASONS = ['Consultation','Checkup','Emergency','Therapy','Follow-up','Other']

def load(path, cols):
    if not os.path.exists(path): pd.DataFrame(columns=cols).to_csv(path,index=False)
    try: df = pd.read_csv(path,dtype=str).fillna('')
    except Exception: df = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns: df[c] = ''
    return df[cols]

def save(df,path): df.to_csv(path,index=False)
def patients(): return load(PF,PC)
def doctors(): return load(DF,DC)
def appointments(): return load(AF,AC)
def new_id(df,col,prefix):
    nums=[]
    for x in df[col].astype(str):
        if x.startswith(prefix):
            try: nums.append(int(x[len(prefix):]))
            except: pass
    return f'{prefix}{max(nums,default=0)+1:03d}'

def dashboard():
    a=appointments(); p=patients(); d=doctors()
    counts=a.status.value_counts().reindex(STATUSES,fill_value=0)
    msg=(f'## Hospital Management Dashboard\n**Patients:** {len(p)} | **Doctors:** {len(d)} | **Appointments:** {len(a)}\n\n'
         f'**Scheduled:** {counts["Scheduled"]} | **Completed:** {counts["Completed"]} | **Cancelled:** {counts["Cancelled"]} | **No-show:** {counts["No-show"]}')
    fig=px.pie(names=counts.index,values=counts.values,title='Appointment Status Distribution',hole=.35)
    return msg,fig,a.tail(10)

def add_patient(pid,name,age,gender,phone,address):
    df=patients()
    if not name.strip(): return 'Please enter patient name.',df
    pid=pid.strip() or new_id(df,'patient_id','P')
    if pid in df.patient_id.values: return 'Patient ID already exists.',df
    row=[pid,name.strip(),str(age or ''),gender,phone.strip(),address.strip()]
    df.loc[len(df)]=row; save(df,PF)
    return f'Patient {pid} added successfully.',df

def add_doctor(did,name,special,phone,availability):
    df=doctors()
    if not name.strip(): return 'Please enter doctor name.',df
    did=did.strip() or new_id(df,'doctor_id','D')
    if did in df.doctor_id.values: return 'Doctor ID already exists.',df
    df.loc[len(df)]=[did,name.strip(),special.strip(),phone.strip(),availability]; save(df,DF)
    return f'Doctor {did} added successfully.',df

def book(pid,did,date,time,reason):
    df=appointments()
    if pid not in patients().patient_id.values: return 'Select a valid patient.',df
    if did not in doctors().doctor_id.values: return 'Select a valid doctor.',df
    if not date or not time: return 'Enter appointment date and time.',df
    aid=new_id(df,'appointment_id','A')
    df.loc[len(df)]=[aid,pid,did,date,time,reason,'Scheduled']; save(df,AF)
    return f'Appointment {aid} booked successfully.',df

def search_patient(q):
    df=patients()
    if q.strip(): df=df[df.astype(str).apply(lambda c:c.str.contains(q,case=False,na=False)).any(axis=1)]
    return df

def search_doctor(q):
    df=doctors()
    if q.strip(): df=df[df.astype(str).apply(lambda c:c.str.contains(q,case=False,na=False)).any(axis=1)]
    return df

def search_appointment(q,status):
    df=appointments()
    if q.strip(): df=df[df.astype(str).apply(lambda c:c.str.contains(q,case=False,na=False)).any(axis=1)]
    if status!='All': df=df[df.status==status]
    return df

def update(aid,status):
    df=appointments()
    if aid not in df.appointment_id.values: return 'Appointment ID not found.',df
    df.loc[df.appointment_id==aid,'status']=status; save(df,AF)
    return f'{aid} updated successfully.',df
