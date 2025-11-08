from llama_index.multi_modal_llms import GeminiMultiModal
from llama_index.program import MultiModalLLMCompletionProgram
from llama_index.output_parsers import PydanticOutputParser
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from pydantic import BaseModel, Field
from typing_extensions import Annotated
import os
import base64
import json
import logging
from PIL import Image
from io import BytesIO

try:
    # OpenAI client (used for OpenRouter via OpenAI-compatible API)
    from openai import OpenAI
except Exception:
    OpenAI = None

damages_initial_prompt_str = """
The images are of a damaged {make_name} {model_name} {year} car. 
The images are taken from different angles.
Please analyze them and tell me what parts are damaged and what is the estimated cost of repair.
"""

conditions_report_initial_prompt_str = """
The images are of a damaged vehicle. 
I need to fill a vehicle condition report based on the picture(s).
Please fill the following details based on the image(s):
FRONT
1. Roof
2. Windshield
3. Hood
4. Grill
5. Front bumper
6. Right mirror
7. Left mirror
8. Front right light
9. Front left light
BACK
10. Rear Window
11. Trunk/TGate
12. Trunk/Cargo area
13. Rear bumper
14. Tail lights
DRIVERS SIDE 
15. Left fender
16. Left front door
17. Left rear door
18. Left rear quarter panel
PASSENGER SIDE
19. Right rear quarter
20. Right rear door
21. Right front door
22. Right fender
TIRES
T1. Front left tire
T2. Front right tire
T3. Rear left tire
T4. Rear right tire

For each of the details you must answer with a score based on this descriptions to reflect the condition: 

- 0: Not visible
- 1: Seems OK (no damage)
- 2: Minor damage (scratches, dents)
- 3: Major damage (bent, broken, missing)
"""


class DamagedPart(BaseModel):
    """Data model of the damaged part"""

    part_name: str = Field(..., description="Name of the damaged part")
    cost: float = Field(..., description="Estimated cost of repair")


class DamagedParts(BaseModel):
    """Data model of the damaged parts"""

    damaged_parts: list[DamagedPart] = Field(..., description="List of damaged parts")
    summary: str = Field(..., description="Summary of the damage")


class ConditionsReport(BaseModel):
    """Data model of conditions report"""

    roof: Annotated[int, Field(0, ge=0, le=3, description="Roof condition")]
    windshield: Annotated[int, Field(0, ge=0, le=3, description="Windshield condition")]
    hood: Annotated[int, Field(0, ge=0, le=3, description="Hood condition")]
    grill: Annotated[int, Field(0, ge=0, le=3, description="Grill condition")]
    front_bumper: Annotated[
        int, Field(0, ge=0, le=3, description="Front bumper condition")
    ]
    right_mirror: Annotated[
        int, Field(0, ge=0, le=3, description="Right mirror condition")
    ]
    left_mirror: Annotated[
        int, Field(0, ge=0, le=3, description="Left mirror condition")
    ]
    front_right_light: Annotated[
        int, Field(0, ge=0, le=3, description="Front right light condition")
    ]
    front_left_light: Annotated[
        int, Field(0, ge=0, le=3, description="Front left light condition")
    ]
    # back
    rear_window: Annotated[
        int, Field(0, ge=0, le=3, description="Rear window condition")
    ]
    trunk_tgate: Annotated[
        int, Field(0, ge=0, le=3, description="Trunk/TGate condition")
    ]
    trunk_cargo_area: Annotated[
        int, Field(0, ge=0, le=3, description="Trunk/Cargo area condition")
    ]
    rear_bumper: Annotated[
        int, Field(0, ge=0, le=3, description="Rear bumper condition")
    ]
    right_tail_light: Annotated[
        int, Field(0, ge=0, le=3, description="Right tail light condition")
    ]
    left_tail_light: Annotated[
        int, Field(0, ge=0, le=3, description="Left tail light condition")
    ]
    # left
    left_rear_quarter: Annotated[
        int, Field(0, ge=0, le=3, description="Left rear quarter condition")
    ]
    left_rear_door: Annotated[
        int, Field(0, ge=0, le=3, description="Left rear door condition")
    ]
    left_front_door: Annotated[
        int, Field(0, ge=0, le=3, description="Left front door condition")
    ]
    left_fender: Annotated[
        int, Field(0, ge=0, le=3, description="Left fender condition")
    ]
    left_front_tire: Annotated[
        int, Field(0, ge=0, le=3, description="Left front tire condition")
    ]
    left_rear_tire: Annotated[
        int, Field(0, ge=0, le=3, description="Left rear tire condition")
    ]
    # right
    right_rear_quarter: Annotated[
        int, Field(0, ge=0, le=3, description="Right rear quarter condition")
    ]
    right_rear_door: Annotated[
        int, Field(0, ge=0, le=3, description="Right rear door condition")
    ]
    right_front_door: Annotated[
        int, Field(0, ge=0, le=3, description="Right front door condition")
    ]
    right_fender: Annotated[
        int, Field(0, ge=0, le=3, description="Right fender condition")
    ]
    right_front_tire: Annotated[
        int, Field(0, ge=0, le=3, description="Right front tire condition")
    ]
    right_rear_tire: Annotated[
        int, Field(0, ge=0, le=3, description="Right rear tire condition")
    ]


def pydantic_llm(
    output_class, image_documents, prompt_template_str, selected_llm_model
):
    # Lazily instantiate the selected LLM to avoid requiring credentials for unused providers
    if selected_llm_model == "OpenAI":
        multi_modal_llm = OpenAIMultiModal(model="gpt-4-vision-preview")
    elif selected_llm_model == "OpenRouter":
        # For OpenRouter, use the OpenAI-compatible client directly to avoid
        # OpenAIMultiModal's hardcoded model validation.
        multi_modal_llm = None
    else:  # Gemini
        multi_modal_llm = GeminiMultiModal(model_name="models/gemini-pro-vision")

    # OpenRouter direct path
    if selected_llm_model == "OpenRouter":
        if OpenAI is None:
            raise RuntimeError(
                "OpenAI client not available. Please ensure the 'openai' package is installed."
            )

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key or not base_url:
            raise RuntimeError(
                "Missing OPENAI_API_KEY or OPENAI_BASE_URL in environment for OpenRouter."
            )

        client = OpenAI(api_key=api_key, base_url=base_url)

        # Build content: initial instructions + images as base64 data URLs
        contents = [{"type": "text", "text": prompt_template_str}]

        # Convert image documents to base64 data URLs with optimization
        for doc in image_documents:
            image_path = None
            # Try common attributes for local path
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                image_path = doc.metadata.get("file_path") or doc.metadata.get("path")
            if not image_path:
                for attr in ("id_", "doc_id", "source_path"):
                    if hasattr(doc, attr):
                        candidate = getattr(doc, attr)
                        if isinstance(candidate, str) and os.path.exists(candidate):
                            image_path = candidate
                            break
            if not image_path and hasattr(doc, "text") and isinstance(doc.text, str):
                maybe = doc.text.strip()
                if os.path.exists(maybe):
                    image_path = maybe

            if not image_path:
                continue

            try:
                # Downscale and compress to reduce prompt token load
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    img.thumbnail((1024, 1024))
                    buf = BytesIO()
                    img.save(buf, format="JPEG", quality=85, optimize=True)
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode("utf-8")
                mime = "image/jpeg"
                contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            except Exception:
                continue

        system_prompt = (
            "You are an expert auto damage estimator. "
            "Analyze provided vehicle images and respond ONLY with JSON. "
            "Each condition field must be an integer: 0 (not visible), 1 (OK), 2 (minor), 3 (major)."
        )

        # Provide a JSON schema hint to steer formatting strictly
        schema_hint = (
            "Return a strictly valid JSON object with these required fields: "
            "roof, windshield, hood, grill, front_bumper, right_mirror, left_mirror, "
            "front_right_light, front_left_light, rear_window, trunk_tgate, trunk_cargo_area, "
            "rear_bumper, right_tail_light, left_tail_light, left_rear_quarter, left_rear_door, "
            "left_front_door, left_fender, left_front_tire, left_rear_tire, right_rear_quarter, "
            "right_rear_door, right_front_door, right_fender, right_front_tire, right_rear_tire."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": contents + [{"type": "text", "text": schema_hint}]},
        ]

        completion = client.chat.completions.create(
            model="google/gemini-2.5-pro",
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
            # Set a high upper bound; provider caps to model limits and only uses what's needed
            max_tokens=80000,
        )

        # Prefer parsed JSON if available (OpenAI SDK v2 provides .parsed when using response_format)
        msg = completion.choices[0].message
        if hasattr(msg, "parsed") and isinstance(msg.parsed, (dict, list)):
            parsed_obj = msg.parsed
            if isinstance(parsed_obj, list):
                # Some providers may return a list-wrapped object
                try:
                    parsed_obj = parsed_obj[0]
                except Exception:
                    pass
            try:
                validated = output_class.model_validate(parsed_obj)
                return validated, {"fallback_used": False}
            except Exception:
                # Fall through to text-based parsing
                pass

        # Fallback to text content. Content may be a string or a list of segments.
        content = getattr(msg, "content", "")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # Extract text fields from content array
            parts = []
            for item in content:
                try:
                    if isinstance(item, dict):
                        # OpenAI-style: {type: 'text', text: '...'}
                        if item.get("type") == "text" and isinstance(item.get("text"), str):
                            parts.append(item["text"]) 
                        # Some providers use {type: 'output_text', text: '...'}
                        elif item.get("type") == "output_text" and isinstance(item.get("text"), str):
                            parts.append(item["text"]) 
                    # SDK object with .text
                    elif hasattr(item, "text"):
                        parts.append(getattr(item, "text"))
                except Exception:
                    continue
            text = "\n".join([p for p in parts if isinstance(p, str)])
        else:
            text = str(content or "")

        def _try_parse_json(raw: str):
            # First attempt: direct parse
            try:
                return json.loads(raw)
            except Exception:
                pass
            # Second: strip code fences
            s = raw.strip()
            if s.startswith("```") and s.endswith("```"):
                s = s.strip("`\n")
                try:
                    return json.loads(s)
                except Exception:
                    pass
            # Third: extract the first JSON object substring
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(s[start : end + 1])
                except Exception:
                    pass
            return None

        data = None
        if text.strip():
            data = _try_parse_json(text)

        used_fallback = False
        if data is None:
            used_fallback = True
            logging.warning(
                "Empty or non-JSON response from provider; applying zeroed fallback ConditionsReport."
            )
            # Build a safe fallback with zeros for all integer fields
            try:
                data = {key: 0 for key in output_class.model_fields.keys()}
            except Exception:
                data = {}
        validated = output_class.model_validate(data)
        return validated, {"fallback_used": used_fallback}

    # Default path: use LlamaIndex program for OpenAI or Gemini
    llm_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_class),
        image_documents=image_documents,
        prompt_template_str=prompt_template_str,
        multi_modal_llm=multi_modal_llm,
        verbose=True,
    )

    response = llm_program()
    return response, {"fallback_used": False}
