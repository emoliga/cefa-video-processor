"""
main.py - Entry point for Render deployment.
Registers both the video processing and docx generation endpoints.
"""

import os
from flask import Flask
from app import app
from docx_endpoint import docx_bp

app.register_blueprint(docx_bp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
