"""Compatibility entrypoint for the existing Render start command.

The Render service was configured for v1 with:
    gunicorn server:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120

Plain gunicorn workers speak WSGI and can't serve a FastAPI (ASGI) app, so this
module wraps the v2 app in an ASGI-to-WSGI adapter. That lets v2 deploy with no
dashboard changes. The preferred entrypoint remains `uvicorn main:app` (see
Procfile); this shim exists only so the old start command keeps working.
"""
from a2wsgi import ASGIMiddleware

from main import app as asgi_app

app = ASGIMiddleware(asgi_app)
