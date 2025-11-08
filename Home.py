import streamlit as st
st.set_page_config(page_title="AutoTraceAi", page_icon="🚗")
from dotenv import load_dotenv
import cv2
import numpy as np
import os
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
from streamlit_modal import Modal
import streamlit.components.v1 as components
import base64
from datetime import datetime

modal = Modal("Damage Report", key="demo", max_width=1280)

# External report service disabled; generating local report HTML instead
api_url = None


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
    # Generate colored images for all sides
    sides = ["front", "back", "left", "right"]
    colored_images = {}
    for side in sides:
        img = process_car_parts(conditions, side)
        colored_images[side] = encode_image_to_data_url(img)

    # Split conditions by side using sides_map from car_colorizer
    from car_colorizer import sides_map

    conditions_map = {s: {} for s in sides}
    for part, cond in conditions.items():
        for side in sides:
            if part in sides_map[side]["parts"]:
                conditions_map[side][part] = cond
                break

    # Build simple HTML string
    def section_html(side: str) -> str:
        items_html = "\n".join(
            [
                f"<li><strong>{p.replace('_',' ').title()}</strong>: {status_text(c)}</li>"
                for p, c in conditions_map[side].items()
            ]
        )
        return (
            f"<section>"
            f"<h3>{side.title()} side</h3>"
            f"<img src=\"{colored_images[side]}\" alt=\"{side} view\" style=\"max-width:100%;height:auto;border:1px solid #ddd\"/>"
            f"<ul>{items_html}</ul>"
            f"</section>"
        )

    html = (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\"/>"
        "<title>Vehicle Condition Report</title>"
        "<style>body{font-family:Arial,sans-serif;padding:16px}section{margin-bottom:24px}</style>"
        "</head><body>"
        f"<h1>Vehicle Condition Report</h1>"
        f"<p><strong>Vehicle:</strong> {car_name}</p>"
        f"{section_html('front')}"
        f"{section_html('back')}"
        f"{section_html('right')}"
        f"{section_html('left')}"
        "</body></html>"
    )
    return html


load_dotenv()


states_names = ["front_image", "back_image", "left_image", "right_image", "report_id"]

openai_mm_llm = OpenAIMultiModal(model="gpt-4-vision-preview")

# Remove form border and padding styles
css = r"""
    <style>
        [data-testid="stForm"] {border: 0px;padding:0px}
    </style>
"""
st.markdown(css, unsafe_allow_html=True)


for state_name in states_names:
    if state_name not in st.session_state:
        st.session_state[state_name] = None


st.title("AutoTraceAi")


st.subheader("Upload your car crash pictures")


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


col1, col2 = st.columns(2)

with col1:
    create_drag_and_drop("front_image", "Front Image")
    create_drag_and_drop("right_image", "Left Image")

with col2:
    create_drag_and_drop("back_image", "Back Image")
    create_drag_and_drop("left_image", "Right Image")


def save_image(state_name):
    path = os.path.join(os.getcwd(), "uploads")
    if not os.path.exists(path):
        os.makedirs(path)

    if st.session_state[state_name] is not None:
        with open(os.path.join(path, f"{state_name}.jpg"), "wb") as f:
            f.write(st.session_state[state_name].getbuffer())


def delete_image(state_name):
    # Delete from the uploads directory where files were saved
    path = os.path.join(os.getcwd(), "uploads")
    file_path = os.path.join(path, f"{state_name}.jpg")
    if st.session_state[state_name] is not None and os.path.exists(file_path):
        os.remove(file_path)


with st.form(key="car_form"):
    selected_make = st.selectbox(
        "Select your car make",
        ("Ford", "Subaru", "BMW", "Mercedes", "Volkswagen", "Volvo"),
    )

    selected_model = st.selectbox(
        "Select your car model",
        ("Mustang", "Outback", "X3", "C-Class", "Golf", "XC60"),
    )

    selected_year = st.selectbox(
        "Select your car year",
        ("2007", "2010", "2011", "2012", "2013", "2014"),
    )

    selected_llm_model = st.selectbox(
        "Select LLM model",
        ("Gemini", "OpenAI", "OpenRouter"),
    )

    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    with st.spinner("Processing..."):
        for state_name in states_names:
            save_image(state_name)
        # Read uploaded images from the dedicated uploads directory
        path = os.path.join(os.getcwd(), "uploads")

        # Guard: ensure there are image files before reading directory
        try:
            uploaded_files = [
                f
                for f in os.listdir(path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        except FileNotFoundError:
            uploaded_files = []

        if not uploaded_files:
            st.warning(
                "No images found in 'uploads/'. Please upload images and try again.",
                icon="⚠️",
            )
            # Stop execution to avoid ValueError from SimpleDirectoryReader
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

        request_data = []

        for part, condition in dict(conditions_report_response).items():
            request_data.append({"part": part, "condition": condition})

        # Build local report HTML instead of calling remote service
        car_name = f"{selected_make} {selected_model} {selected_year}"
        local_report_html = build_local_report_html(
            dict(conditions_report_response), car_name
        )
        st.session_state["local_report_html"] = local_report_html

        if llm_meta.get("fallback_used"):
            st.info(
                "The model returned an empty or non-JSON response. Using a safe fallback; S3 upload skipped.",
                icon="ℹ️",
            )
        # Bypass S3 uploads entirely for local flow

        modal.open()

if modal.is_open() and st.session_state.get("local_report_html"):
    with modal.container():
        # Show local report inline
        components.html(
            st.session_state["local_report_html"], height=500, scrolling=True
        )

        # Offer a download of the report HTML
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="Download report as HTML",
            data=st.session_state["local_report_html"],
            file_name=f"vehicle_report_{ts}.html",
            mime="text/html",
        )

        # st.subheader("Summary")
        # st.write(damages_response.summary)

        # st.subheader("Damaged Parts")
        # df = pd.DataFrame.from_records(
        #     [part.model_dump() for part in damages_response.damaged_parts]
        # )
        # st.dataframe(df)

        # TODO: look for the parts in the vector store

        # filters = MetadataFilters(
        #     filters=[
        #         MetadataFilter(key="make", value=selected_make),
        #         MetadataFilter(key="model", value=selected_model),
        #         MetadataFilter(key="year", value=selected_year),
        #     ]
        # )

        # retriever = VectorStoreIndex.from_vector_store(vector_store).as_retriever(
        #     filters=filters,
        # )

        # query_engine = RetrieverQueryEngine(
        #     retriever=retriever,
        # )
