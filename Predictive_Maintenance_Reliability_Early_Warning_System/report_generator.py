"""PDF Executive Report Generator."""

from reportlab.pdfgen import canvas
from config import REPORT_DIR
import os

def generate_pdf_report(asset_data: dict, filename: str = "Executive_Report.pdf"):
    """Generates a professional PDF summary for maintenance leadership."""
    path = os.path.join(REPORT_DIR, filename)
    c = canvas.Canvas(path)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "DCP Predictive Maintenance - Executive Report")
    c.setFont("Helvetica", 12)
    
    y = 700
    for key, value in asset_data.items():
        c.drawString(100, y, f"{key}: {value}")
        y -= 20
        
    c.drawString(100, y - 50, "Signed: __________________________")
    c.save()
    return path