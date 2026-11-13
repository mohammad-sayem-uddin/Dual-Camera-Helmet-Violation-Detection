"""Temporary local blind human ground-truth review dashboard for dual_camera_gt_v1.1_blind_human."""
from __future__ import annotations

DASHBOARD_VERSION = "1.1.0-review-dashboard-1"

from src.review_dashboard.config import DashboardConfig  # noqa: E402,F401

__all__ = ["DASHBOARD_VERSION", "DashboardConfig"]
