import streamlit as st
st.set_page_config(page_title="AutoTraceAi", page_icon="🚗", layout="wide", initial_sidebar_state="collapsed")
from dotenv import load_dotenv
import cv2
import numpy as np
import os
import json
from llama_index import SimpleDirectoryReader
from pydantic_llm import (
    pydantic_llm,
    DamagedParts,
    damages_initial_prompt_str,
    ConditionsReport,
    conditions_report_initial_prompt_str,
)
import pandas as pd
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from car_colorizer import process_car_parts
import requests
from io import BytesIO
import streamlit.components.v1 as components
import base64
from datetime import datetime

# Import db and auth
from database import SessionLocal, User, Report
from auth import authenticate_user, create_user

load_dotenv()

def encode_image_to_data_url(img):
    in_memory_file = BytesIO()
    img.save(in_memory_file, format="PNG")
    in_memory_file.seek(0)
    b64 = base64.b64encode(in_memory_file.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def status_text(condition: int) -> str:
    mapping = {
        0: "Not visible",
        1: "Seems OK",
        2: "Minor damage",
        3: "Major damage",
    }
    return mapping.get(condition, "Unknown")

def build_local_report_html(conditions: dict, car_name: str) -> str:
    sides = ["front", "back", "left", "right"]
    colored_images = {}
    for side in sides:
        img = process_car_parts(conditions, side)
        colored_images[side] = encode_image_to_data_url(img)

    from car_colorizer import sides_map

    conditions_map = {s: {} for s in sides}
    for part, cond in conditions.items():
        for side in sides:
            if part in sides_map[side]["parts"]:
                conditions_map[side][part] = cond
                break

    def section_html(side: str) -> str:
        items_html = "\n".join(
            [
                f"<li class='py-3 flex justify-between border-b border-gray-200 last:border-0'><span class='font-medium text-gray-700'>{p.replace('_',' ').title()}</span> <span class='text-gray-900'>{status_text(c)}</span></li>"
                for p, c in conditions_map[side].items()
            ]
        )
        return (
            f"<div class='bg-white shadow rounded-lg p-6 mb-6'>"
            f"<h3 class='text-xl font-bold text-gray-800 mb-4 capitalize'>{side} Side</h3>"
            f"<div class='flex flex-col md:flex-row gap-6'>"
            f"<div class='w-full md:w-1/2 flex items-center justify-center bg-gray-50 p-4 rounded-lg'><img src=\"{colored_images[side]}\" alt=\"{side} view\" class='max-w-full h-auto object-contain'/></div>"
            f"<div class='w-full md:w-1/2'><ul class='divide-y divide-gray-200'>{items_html}</ul></div>"
            f"</div>"
            f"</div>"
        )

    html = (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\"/>"
        "<title>Vehicle Condition Report</title>"
        "<script src=\"https://cdn.tailwindcss.com\"></script>"
        "</head><body class='bg-gray-100 p-8 font-sans'>"
        f"<div class='max-w-4xl mx-auto'>"
        f"<div class='bg-blue-600 text-white p-6 rounded-t-lg shadow-md mb-6'>"
        f"<h1 class='text-3xl font-extrabold tracking-tight'>Vehicle Condition Report</h1>"
        f"<p class='text-lg mt-2 font-medium'>Vehicle: {car_name}</p>"
        f"</div>"
        f"{section_html('front')}"
        f"{section_html('back')}"
        f"{section_html('right')}"
        f"{section_html('left')}"
        "</div>"
        "</body></html>"
    )
    return html

states_names = ["front_image", "back_image", "left_image", "right_image", "report_id"]
openai_mm_llm = OpenAIMultiModal(model="gpt-4-vision-preview")

# ----------------- UI STYLING -----------------
def inject_custom_css():
    css = """
    <style>
        /* Hide sidebar toggle and default sidebar padding entirely */
        [data-testid="collapsedControl"] {
            display: none;
        }
        
        section[data-testid="stSidebar"] {
            width: 0px !important;
        }

        /* Remove form borders */
        [data-testid="stForm"] {
            border: none;
            padding: 0;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Inject styling globally
inject_custom_css()

# ----------------- INIT SESSION STATE -----------------
for state_name in states_names:
    if state_name not in st.session_state:
        st.session_state[state_name] = None

if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "show_new_report" not in st.session_state:
    st.session_state["show_new_report"] = False

# ----------------- APP LOGIC -----------------
def save_image(state_name):
    path = os.path.join(os.getcwd(), "uploads")
    if not os.path.exists(path):
        os.makedirs(path)

    if st.session_state[state_name] is not None:
        with open(os.path.join(path, f"{state_name}.jpg"), "wb") as f:
            f.write(st.session_state[state_name].getbuffer())

def delete_image(state_name):
    path = os.path.join(os.getcwd(), "uploads")
    file_path = os.path.join(path, f"{state_name}.jpg")
    if st.session_state[state_name] is not None and os.path.exists(file_path):
        os.remove(file_path)

def create_drag_and_drop(state_name, label):
    st.session_state[state_name] = st.file_uploader(
        label=label, key=f"{state_name}_image"
    )

    if st.session_state[state_name] is not None:
        css = f"""
            <style>
                [aria-label="{label}"] {{display: none;}}
            </style>
        """
        st.markdown(css, unsafe_allow_html=True)
        file_bytes = np.asarray(
            bytearray(st.session_state[state_name].read()), dtype=np.uint8
        )
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels="BGR")

def login_signup_ui():
    st.title("AutoTraceAi")
    st.markdown("### AI Vehicle Assessment")
    
    # Center the login box using columns
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("login_form"):
                st.markdown("### Welcome Back")
                username = st.text_input("Email or Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True)
                if submitted:
                    if not username or not password:
                        st.error("Please enter your credentials")
                    else:
                        try:
                            db = SessionLocal()
                            user = authenticate_user(db, username, password)
                            db.close()
                            if user:
                                st.session_state["user_id"] = user.id
                                st.session_state["username"] = user.username
                                st.rerun()
                            else:
                                st.error("Invalid credentials")
                        except Exception as e:
                            st.error(f"Database error: {e}. Make sure DATABASE_URL is set correctly in .env!")
                        
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("signup_form"):
                st.markdown("### Create an Account")
                new_email = st.text_input("Email Address")
                new_username = st.text_input("Choose Username")
                new_password = st.text_input("Choose Password", type="password")
                submitted = st.form_submit_button("Sign Up", use_container_width=True)
                if submitted:
                    if not new_email or not new_username or not new_password:
                        st.error("Please fill all fields")
                    else:
                        try:
                            db = SessionLocal()
                            user = create_user(db, new_username, new_email, new_password)
                            db.close()
                            if user:
                                st.success("Account created successfully! Please switch to the Login tab.")
                            else:
                                st.error("Username or Email already exists")
                        except Exception as e:
                            st.error(f"Database error: {e}. Make sure DATABASE_URL is set correctly in .env!")

def main_app_ui():
    st.title("AutoTraceAi")
    
    col1, col2 = st.columns([8, 2])
    with col1:
        st.write(f"Welcome, **{st.session_state['username']}**")
    with col2:
        if st.button("Logout", use_container_width=True):
            st.session_state["user_id"] = None
            st.session_state["username"] = None
            st.rerun()
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    tab_scan, tab_history = st.tabs(["🔍 New Scan", "📜 Report History"])
    
    with tab_scan:
        st.markdown("### Upload your car crash pictures")
        st.markdown("Please upload pictures from all four sides for an accurate assessment.")
        
        col1, col2 = st.columns(2)
        with col1:
            create_drag_and_drop("front_image", "Front Image")
            create_drag_and_drop("right_image", "Left Image")
        with col2:
            create_drag_and_drop("back_image", "Back Image")
            create_drag_and_drop("left_image", "Right Image")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### Vehicle Details")
        with st.form(key="car_form"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                selected_make = st.selectbox("Make", ("Ford", "Subaru", "BMW", "Mercedes", "Volkswagen", "Volvo"))
            with col_b:
                selected_model = st.selectbox("Model", ("Mustang", "Outback", "X3", "C-Class", "Golf", "XC60"))
            with col_c:
                selected_year = st.selectbox("Year", ("2007", "2010", "2011", "2012", "2013", "2014"))
            
            selected_llm_model = "OpenRouter"
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button(label="Generate Assessment Report", use_container_width=True)

        if submit_button:
            with st.spinner("Analyzing images and generating report..."):
                for state_name in states_names:
                    save_image(state_name)
                
                path = os.path.join(os.getcwd(), "uploads")
                try:
                    uploaded_files = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                except FileNotFoundError:
                    uploaded_files = []

                if not uploaded_files:
                    st.warning("No images found. Please upload images and try again.", icon="⚠️")
                    st.stop()

                image_documents = SimpleDirectoryReader(path).load_data()
                conditions_report_response, llm_meta = pydantic_llm(
                    output_class=ConditionsReport,
                    image_documents=image_documents,
                    prompt_template_str=conditions_report_initial_prompt_str.format(
                        make_name=selected_make, model_name=selected_model, year=selected_year
                    ),
                    selected_llm_model=selected_llm_model,
                )

                for state_name in states_names:
                    delete_image(state_name)

                car_name = f"{selected_make} {selected_model} {selected_year}"
                local_report_html = build_local_report_html(dict(conditions_report_response), car_name)
                st.session_state["local_report_html"] = local_report_html
                st.session_state["show_new_report"] = True

                # Save to database
                try:
                    db = SessionLocal()
                    new_report = Report(
                        user_id=st.session_state["user_id"],
                        car_name=car_name,
                        report_html=local_report_html,
                        conditions_json=json.dumps(dict(conditions_report_response))
                    )
                    db.add(new_report)
                    db.commit()
                    db.close()
                    st.toast("Report saved to history successfully!", icon="✅")
                except Exception as e:
                    st.error(f"Failed to save report to database: {e}")

                if llm_meta.get("fallback_used"):
                    st.info("The model returned an empty or non-JSON response. Using a safe fallback.", icon="ℹ️")

        if st.session_state.get("show_new_report") and st.session_state.get("local_report_html"):
            st.markdown("### Current Assessment Results")
            components.html(st.session_state["local_report_html"], height=800, scrolling=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download Report as HTML",
                data=st.session_state["local_report_html"],
                file_name=f"vehicle_report_{ts}.html",
                mime="text/html",
                use_container_width=True
            )

    with tab_history:
        st.markdown("### Your Past Assessments")
        try:
            db = SessionLocal()
            reports = db.query(Report).filter(Report.user_id == st.session_state["user_id"]).order_by(Report.created_at.desc()).all()
            db.close()
            
            if not reports:
                st.info("No reports found. Generate a new scan to see it here.")
            else:
                for rep in reports:
                    with st.expander(f"{rep.car_name} - {rep.created_at.strftime('%B %d, %Y at %I:%M %p')}"):
                        components.html(rep.report_html, height=450, scrolling=True)
                        st.download_button(
                            label="Download HTML",
                            data=rep.report_html,
                            file_name=f"report_{rep.id}.html",
                            mime="text/html",
                            key=f"dl_{rep.id}",
                            use_container_width=True
                        )
        except Exception as e:
            st.error(f"Failed to fetch reports: {e}")

if st.session_state["user_id"] is None:
    login_signup_ui()
else:
    main_app_ui()
