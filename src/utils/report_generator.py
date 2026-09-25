"""
Clinical PDF Diagnostic Report Generator
=========================================
Generates standardized, physician-ready PDF reports containing patient metadata,
diagnostic classification probability, Youden J cutoff, and Grad-CAM++ heatmap overlays.
"""

import os
import io
import time
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_diagnostic_pdf(
    patient_id: str,
    class_name: str,
    probability: float,
    confidence: float,
    model_name: str,
    output_path: Optional[Path] = None,
    original_img_path: Optional[str] = None,
    overlay_img_path: Optional[str] = None,
    optimal_threshold: float = 0.50,
) -> bytes:
    """
    Generate a 1-click clinical diagnostic PDF report.
    """
    buffer = io.BytesIO()
    target = str(output_path) if output_path else buffer

    if REPORTLAB_AVAILABLE:
        doc = SimpleDocTemplate(target, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#1e1b4b"),
            spaceAfter=6,
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#6b21a8"),
            spaceAfter=14,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#1f2937"),
        )

        elements = []

        # Header
        elements.append(Paragraph("PANCREATIC CANCER AI DIAGNOSTIC REPORT", title_style))
        elements.append(Paragraph(f"Autonomous Clinical Screening System · Generated on {time.strftime('%Y-%m-%d %H:%M:%S UTC')}", subtitle_style))
        elements.append(Spacer(1, 10))

        # Patient & Study Information Table
        patient_data = [
            [Paragraph("<b>Patient Identifier:</b>", body_style), Paragraph(patient_id, body_style),
             Paragraph("<b>Imaging Modality:</b>", body_style), Paragraph("Abdominal CT", body_style)],
            [Paragraph("<b>Target Organ:</b>", body_style), Paragraph("Pancreas (Parenchyma)", body_style),
             Paragraph("<b>Vision Backbone:</b>", body_style), Paragraph(model_name.upper(), body_style)],
            [Paragraph("<b>Diagnostic Threshold:</b>", body_style), Paragraph(f"{optimal_threshold:.2f} (Youden's J)", body_style),
             Paragraph("<b>Review Status:</b>", body_style), Paragraph("Preliminary AI Screening", body_style)],
        ]
        t_patient = Table(patient_data, colWidths=[120, 150, 120, 150])
        t_patient.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f3ff")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#ddd6fe")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ede9fe")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(t_patient)
        elements.append(Spacer(1, 16))

        # Classification Summary
        is_cancer = class_name.lower() == "cancerous"
        result_color = "#dc2626" if is_cancer else "#16a34a"
        result_text = "POSITIVE FOR MALIGNANT LESION" if is_cancer else "NEGATIVE (NORMAL PARENCHYMA)"

        result_data = [
            [Paragraph(f"<font color='{result_color}' size='13'><b>DIAGNOSTIC FINDING: {result_text}</b></font>", body_style)],
            [Paragraph(f"Malignancy Probability: <b>{probability*100:.2f}%</b> &nbsp;|&nbsp; Diagnostic Confidence: <b>{confidence*100:.2f}%</b>", body_style)],
        ]
        t_result = Table(result_data, colWidths=[540])
        t_result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2" if is_cancer else "#f0fdf4")),
            ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor(result_color)),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(t_result)
        elements.append(Spacer(1, 16))

        # Image Visualizations
        if original_img_path and Path(original_img_path).exists() and overlay_img_path and Path(overlay_img_path).exists():
            img_table_data = [
                [Paragraph("<b>Original CT Slice (W:350 L:45)</b>", body_style), Paragraph("<b>Grad-CAM++ Tumor Localization</b>", body_style)],
                [RLImage(original_img_path, width=220, height=220), RLImage(overlay_img_path, width=220, height=220)],
            ]
            t_img = Table(img_table_data, colWidths=[260, 260])
            t_img.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(t_img)
            elements.append(Spacer(1, 14))

        # Medical Disclaimer
        disclaimer = (
            "<b>Clinical Disclaimer:</b> This report is generated by a computer-aided diagnostic (CAD) neural "
            "network designed for clinical triage support. Definitive diagnosis must be confirmed by a board-certified "
            "radiologist in conjunction with clinical history, biopsy pathology, and comprehensive contrast CT imaging."
        )
        elements.append(Paragraph(disclaimer, ParagraphStyle("Disc", parent=body_style, fontSize=8, textColor=colors.HexColor("#6b7280"))))
        elements.append(Spacer(1, 16))
        elements.append(Paragraph("Physician / Radiologist Reviewer Signature: ____________________________________", body_style))

        doc.build(elements)
    else:
        # Fallback raw byte generation if reportlab is not installed
        pdf_content = (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF\n"
        )
        if output_path:
            with open(output_path, "wb") as f:
                f.write(pdf_content)
        buffer.write(pdf_content)

    return buffer.getvalue()
