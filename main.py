import os
from app import app
from docx_endpoint import docx_bp

app.register_blueprint(docx_bp)

from waitress import serve

port = int(os.environ.get("PORT", 10000))
print(f"Starting waitress on port {port}", flush=True)
serve(app, host="0.0.0.0", port=port, threads=4, channel_timeout=600)
