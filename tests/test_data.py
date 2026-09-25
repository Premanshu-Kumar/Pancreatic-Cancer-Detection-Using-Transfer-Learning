"""
Tests for Medical Data Pipeline, Stratification, and Quality Control
===================================================================
"""

from pathlib import Path
import numpy as np
import pytest

from src.data.split_data import extract_patient_id, split_by_patient_group
from src.data.augmentation import get_augmentation_config, apply_medical_augmentation


class TestDataStratification:
    """Test suite for patient-level grouped splitting and leakage prevention."""

    def test_extract_patient_id_formats(self):
        """Verify regex extraction across various hospital naming conventions."""
        assert extract_patient_id("patient_014_slice_003.png") == "patient_014"
        assert extract_patient_id("P105_slice02.jpg") == "p105"
        assert extract_patient_id("Case_042_axial.tif") == "case_042"

    def test_split_by_patient_group_zero_leakage(self):
        """Ensure slices from the same patient NEVER cross train/val/test splits."""
        records = []
        for p in range(1, 11):
            pid = f"patient_{p:03d}"
            cname = "Cancerous" if p % 2 == 0 else "Normal"
            for s in range(1, 5):
                records.append({
                    "filepath": f"/fake/{pid}_slice_{s}.png",
                    "filename": f"{pid}_slice_{s}.png",
                    "patient_id": pid,
                    "label": "1" if cname == "Cancerous" else "0",
                    "class_name": cname,
                })

        train_r, val_r, test_r = split_by_patient_group(records, 0.6, 0.2, 0.2, seed=42)

        train_pts = set(r["patient_id"] for r in train_r)
        val_pts = set(r["patient_id"] for r in val_r)
        test_pts = set(r["patient_id"] for r in test_r)

        assert len(train_pts.intersection(val_pts)) == 0, "Patient leakage detected between train and val"
        assert len(train_pts.intersection(test_pts)) == 0, "Patient leakage detected between train and test"
        assert len(val_pts.intersection(test_pts)) == 0, "Patient leakage detected between val and test"


class TestMedicalAugmentationConstraints:
    """Verify medical imaging safety constraints on data transforms."""

    def test_vertical_flip_disabled(self):
        """Anatomy constraint: vertical flip must be False to avoid upside-down organs."""
        cfg = get_augmentation_config()
        assert cfg["vertical_flip"] is False

    def test_rotation_range_constrained(self):
        """Rotation must be bounded within +-15 degrees for CT alignment."""
        cfg = get_augmentation_config({"rotation_range": 45})
        assert cfg["rotation_range"] <= 15
