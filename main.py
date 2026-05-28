"""
main.py - Entry point using waitress (no signal conflicts with subprocesses).
"""
import os
from app import app
from docx_endpoint import docx_bp

app.register_blueprint(docx_bp)

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting waitress on port {port}")
    serve(app, host="0.0.0.0", port=port, threads=4, channel_timeout=600)
