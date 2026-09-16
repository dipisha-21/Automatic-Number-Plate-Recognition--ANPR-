"""Vercel-compatible Flask entrypoint.

The application implementation lives in web_app.py; exposing `app` from
app.py lets Vercel's zero-configuration Flask detection find it reliably.
"""
from web_app import app

__all__ = ["app"]
