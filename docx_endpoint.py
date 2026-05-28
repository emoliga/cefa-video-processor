"""
Endpoint to generate a .docx report from processed video steps.
Add this to app.py or keep as a separate blueprint.
"""

import io
import base64
from flask import Blueprint, request, jsonify, send_file
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

docx_bp = Blueprint("docx", __name__)

CEFA_BLUE = RGBColor(0x01, 0x27, 0x7A)
CEFA_RED = RGBColor(0xF0, 0x19, 0x28)


def set_cell_background(cell, hex_color: str):
    """Set table cell background color."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def add_horizontal_rule(doc, color_hex="01277A"):
    """Add a colored horizontal line via paragraph border."""
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), color_hex)
    pBdr.append(bottom)
    pPr.append(pBdr)
    p.paragraph_format.space_after = Pt(6)
    return p


def build_docx(title: str, steps: list[dict], metadata: dict) -> bytes:
    """
    Build the Word document from processed steps.
    Each step: { step_number, start_time, end_time, text, frames_base64 }
    """
    doc = Document()
    
    # --- Page setup (A4) ---
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    
    # --- Styles ---
    styles = doc.styles
    
    # Normal style
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(11)
    
    # Heading 1 override
    h1 = styles["Heading 1"]
    h1.font.name = "Arial"
    h1.font.size = Pt(18)
    h1.font.bold = True
    h1.font.color.rgb = CEFA_BLUE
    
    # Heading 2 override
    h2 = styles["Heading 2"]
    h2.font.name = "Arial"
    h2.font.size = Pt(13)
    h2.font.bold = True
    h2.font.color.rgb = CEFA_BLUE
    
    # --- Cover block ---
    # Title
    title_para = doc.add_paragraph()
    title_para.paragraph_format.space_before = Pt(24)
    title_para.paragraph_format.space_after = Pt(6)
    title_run = title_para.add_run(title)
    title_run.font.name = "Arial"
    title_run.font.size = Pt(22)
    title_run.font.bold = True
    title_run.font.color.rgb = CEFA_BLUE
    title_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    
    add_horizontal_rule(doc, "01277A")
    
    # Metadata line
    date_str = datetime.datetime.now().strftime("%d/%m/%Y")
    meta_para = doc.add_paragraph()
    meta_para.paragraph_format.space_after = Pt(18)
    meta_run = meta_para.add_run(
        f"Generado automáticamente  ·  {date_str}  ·  {metadata.get('total_steps', len(steps))} pasos  ·  Duración: {_fmt_duration(metadata.get('duration', 0))}"
    )
    meta_run.font.name = "Arial"
    meta_run.font.size = Pt(9)
    meta_run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
    
    # Full transcript (collapsed/summary)
    transcript = metadata.get("full_transcript", "")
    if transcript:
        doc.add_heading("Transcripción completa", level=2)
        t_para = doc.add_paragraph(transcript)
        t_para.paragraph_format.space_after = Pt(18)
        for run in t_para.runs:
            run.font.size = Pt(10)
            run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
        add_horizontal_rule(doc, "CCCCCC")
    
    # --- Steps ---
    doc.add_heading("Pasos del procedimiento", level=1)
    
    for step in steps:
        num = step.get("step_number", "?")
        text = step.get("text", "").strip()
        frames = step.get("frames_base64", [])
        start = step.get("start_time", 0)
        end = step.get("end_time", 0)
        
        # Step heading
        step_heading = doc.add_heading(f"Paso {num}", level=2)
        step_heading.paragraph_format.space_before = Pt(14)
        
        # Time badge
        time_para = doc.add_paragraph()
        time_run = time_para.add_run(f"⏱  {_fmt_duration(start)} – {_fmt_duration(end)}")
        time_run.font.size = Pt(9)
        time_run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
        time_para.paragraph_format.space_after = Pt(4)
        
        # Step text
        if text:
            text_para = doc.add_paragraph(text)
            text_para.paragraph_format.space_after = Pt(8)
            for run in text_para.runs:
                run.font.name = "Arial"
                run.font.size = Pt(11)
        
        # Frames: lay them out in a row (max 3 per row)
        if frames:
            _add_frames_row(doc, frames)
        
        # Subtle separator
        add_horizontal_rule(doc, "DDDDDD")
    
    # --- Footer ---
    footer = section.footer
    footer_para = footer.paragraphs[0]
    footer_para.clear()
    footer_run = footer_para.add_run("CEFA Celulosa Fabril · Transformación Digital · Generado automáticamente")
    footer_run.font.name = "Arial"
    footer_run.font.size = Pt(8)
    footer_run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)
    footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Serialize to bytes
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()


def _add_frames_row(doc: Document, frames_b64: list, max_width_cm: float = 5.0):
    """Add images side by side in a table row."""
    # Filter valid frames first
    valid_frames = []
    for f in frames_b64[:3]:
        try:
            img_bytes = base64.b64decode(f)
            if len(img_bytes) > 100:  # sanity check
                valid_frames.append(img_bytes)
        except Exception:
            pass

    n = len(valid_frames)
    if n == 0:
        return

    table = doc.add_table(rows=1, cols=n)
    table.style = "Table Grid"

    row = table.rows[0]
    for i, img_bytes in enumerate(valid_frames):
        cell = row.cells[i]
        para = cell.paragraphs[0]
        para.clear()
        try:
            img_stream = io.BytesIO(img_bytes)
            run = para.add_run()
            run.add_picture(img_stream, width=Cm(max_width_cm))
        except Exception as e:
            para.add_run(f"[imagen {i+1}]")

    doc.add_paragraph()


def _fmt_duration(seconds: float) -> str:
    """Format seconds as mm:ss."""
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


@docx_bp.route("/generate-docx", methods=["POST"])
def generate_docx():
    """
    Expects JSON:
    {
        "title": "Procedimiento: Preparación máquina X",
        "steps": [...],           // from /process-video
        "duration": 180.0,
        "full_transcript": "...",
        "total_steps": 5
    }
    Returns: .docx file as binary (application/vnd.openxmlformats...)
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400
    
    title = data.get("title", "Informe de procedimiento")
    steps = data.get("steps", [])
    metadata = {
        "duration": data.get("duration", 0),
        "full_transcript": data.get("full_transcript", ""),
        "total_steps": data.get("total_steps", len(steps))
    }
    
    try:
        docx_bytes = build_docx(title, steps, metadata)
    except Exception as e:
        return jsonify({"error": f"docx generation failed: {str(e)}"}), 500
    
    # Return as base64 so n8n can handle it easily
    docx_b64 = base64.b64encode(docx_bytes).decode()
    filename = title.replace(" ", "_").replace(":", "")[:60] + ".docx"
    
    return jsonify({
        "docx_base64": docx_b64,
        "filename": filename,
        "size_bytes": len(docx_bytes)
    })
