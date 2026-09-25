"""
Diagnostic Report API Router
============================
Provides 1-click clinical PDF diagnostic report downloads for patients and physicians.
"""

from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import Response

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.report_generator import generate_diagnostic_pdf

router = APIRouter(prefix="/api/report", tags=["report"])


@router.get("/download")
async def download_diagnostic_pdf(
    patient_id: str = Query(default="PATIENT-001"),
    class_name: str = Query(default="Cancerous"),
    probability: float = Query(default=0.92),
    confidence: float = Query(default=0.92),
    model_name: str = Query(default="resnet50"),
    optimal_threshold: float = Query(default=0.50),
):
    """
    Generate and stream a standardized diagnostic PDF report.
    """
    pdf_bytes = generate_diagnostic_pdf(
        patient_id=patient_id,
        class_name=class_name,
        probability=probability,
        confidence=confidence,
        model_name=model_name,
        optimal_threshold=optimal_threshold,
    )

    filename = f"Clinical_Report_{patient_id}_{model_name}.pdf"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": "application/pdf",
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
