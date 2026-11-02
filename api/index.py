"""Vercel Serverless Function entry point.
Exports the FastAPI application instance for Vercel deployment.
"""

import os
import sys

# Ensure repository root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.server.app import app
