# AutoTraceAi — Local Setup (Windows)

This guide walks you through installing Python 3.11, Poetry, dependencies, environment variables, and running the app locally on `http://localhost:8501/`.

## Prerequisites
- Windows 10/11
- Administrator rights (recommended for installs)
- Internet access

## 1) Install Python 3.11
- Download the latest 3.11.x (recommended 3.11.7) from: https://www.python.org/downloads/release/python-3117/
- During installation:
  - Check “Add python.exe to PATH”
  - Optionally install `pip` and the “Python Launcher” if prompted

Verify installation:

```bash
python --version
```

You should see something like: `Python 3.11.7`

## 2) Install Poetry (dependency manager)
Use the official installer in PowerShell:

```bash
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

After installation, restart the terminal so `poetry` is available on PATH.

Verify Poetry:

```bash
poetry --version
```

## 3) Open the project folder
- Ensure you’re in the project root: `d:\GHR\AutoTraceAi`

```bash
cd d:\GHR\AutoTraceAi
```

## 4) Point Poetry to Python 3.11
Tell Poetry to use Python 3.11 for this project:

```bash
poetry env use 3.11
```

If that fails (e.g., multiple Python versions installed), specify the full path to your 3.11 interpreter, for example:

```bash
poetry env use "C:\Users\YourUser\AppData\Local\Programs\Python\Python311\python.exe"
```

## 5) Install project dependencies
Install everything from `pyproject.toml`:

```bash
poetry install
```

## 6) Configure environment variables
Copy the example env file and fill in your keys:

```bash
copy .env.example .env
```

Open `.env` and set values based on the provider you want to use:
- For OpenAI (via LlamaIndex):
  - `OPENAI_API_KEY=sk-...your-openai-key...`
- For OpenRouter (OpenAI-compatible client):
  - `OPENAI_BASE_URL=https://openrouter.ai/api/v1`
  - `OPENAI_API_KEY=sk-or-...your-openrouter-key...`
- Optional:
  - `GOOGLE_API_KEY=...` (if using Gemini)
  - `WEAVIATE_URL=...` and `WEAVIATE_API_KEY=...` (only if you wire up Weaviate)

Note:
- Do not commit `.env` to source control.
- `load_dotenv()` is used in the app to load variables from `.env`.

## 7) Run the app
Start Streamlit via Poetry:

```bash
poetry run streamlit run Home.py
```

Then open the app in your browser:

- http://localhost:8501/

The app’s main page is `Home.py`. The `pages/` directory (e.g., `pages/Example.py`) appears as additional pages in Streamlit’s sidebar.

## 8) Quick command sequence
If you already have Python 3.11 and Poetry:

```bash
poetry env use 3.11
```

```bash
poetry run streamlit run Home.py
```

## 9) Troubleshooting
- Python version mismatch:
  - Confirm `python --version` shows 3.11.x and re-run `poetry env use 3.11`.
- Poetry not found:
  - Make sure you restarted the terminal after installation and that Poetry is on PATH.
- OpenCV errors:
  - On Linux, install system package `libgl1`:
    ```bash
    sudo apt-get update && sudo apt-get install -y libgl1
    ```
  - On Windows, if you see DLL-related errors, ensure Windows is up-to-date. The bundled `opencv-python` wheels typically work out-of-the-box on Windows.
- Missing API keys:
  - Ensure `.env` contains the appropriate keys for your provider (`OPENAI_API_KEY` for OpenAI, or `OPENAI_BASE_URL` and `OPENAI_API_KEY` for OpenRouter).

## 10) What’s included
- Python: `^3.11.7`
- Dependencies (excerpt): `streamlit`, `opencv-python`, `llama-index`, `python-dotenv`, `openai`, `google-generativeai`, `boto3`, `weaviate-client`, `Pillow`, `requests`, `pydantic`, `typing_extensions`, `streamlit-modal`
- Entry point: `Home.py`
- Extra pages: `pages/Example.py`
- Env example: `.env.example`

All set! After completing the steps above, the app should be reachable at `http://localhost:8501/`.