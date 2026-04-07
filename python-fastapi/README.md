# Python - FastAPI

Learning FastAPI from scratch through hands-on exercises.

## Setup

```bash
cd python-fastapi
python -m venv .venv
source .venv/bin/activate  # or venv\Scripts\activate on Windows
or venv\Scripts\activate
pip install fastapi uvicorn
```

## Common Commands

```bash
# Start the server (with auto-reload for development)
uvicorn main:app --reload

# Start on a specific port
uvicorn main:app --reload --port 3000

# Start and allow external access (0.0.0.0 instead of localhost only)
uvicorn main:app --reload --host 0.0.0.0

# Install fastapi with all optional deps (includes uvicorn)
pip install "fastapi[standard]"

# Freeze current dependencies to a file
pip freeze > requirements.txt

# Install from requirements file
pip install -r requirements.txt
```

## FastAPI CLI (requires `fastapi[standard]`)

```bash
# Development mode — auto-reload, localhost only, auto-detects app
fastapi dev main.py

# Production mode — no reload, 0.0.0.0 (external access)
fastapi run main.py
```

Under the hood this is still Uvicorn — just a convenience wrapper with dev-friendly defaults.

## Exercises

| # | Exercise | Description |
|---|----------|-------------|
| 01 | hello-world | Basic app, first endpoint, running uvicorn |

## Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [FastAPI Cheatsheet](https://devsheets.io/sheets/fastapi)
