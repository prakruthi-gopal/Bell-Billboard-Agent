"""
Shared state schema for the Billboard Ad Agent system.
Passed through the pipeline — every agent reads from it and writes back.

Key change: we now track MULTIPLE image assets that the editor
composes together, not just a single base image.
"""

from typing import TypedDict, Optional

class ImageAsset(TypedDict):
    """A single generated image asset with its metadata."""
    role: str           # e.g. "background", "product", "lifestyle"
    path: str           # File path to the image
    description: str    # What this image shows


class BillboardState(TypedDict):
    # --- Input ---
    brief: str

    # --- Guardrail Agent output ---
    guardrail_passed: Optional[bool]    # True if brief is safe and relevant
    guardrail_message: Optional[str]    # Rejection reason if failed

    # --- Planner Agent output ---
    spec: Optional[dict]  # Structured spec: layout, assets needed, text, etc.

    # --- Generator Agent output ---
    image_assets: list[ImageAsset]  # Multiple images generated for composition

    # --- Editor Agent working state ---
    current_image_path: Optional[str]  # Path to the latest composed version
    edit_history: list[str]            # Log of edits with reasoning
    compliance_status: Optional[str]   # "pass" or "fail"
    compliance_violations: list[str]   # What failed
    iteration_count: int               # How many edit loops so far

    # --- Config ---
    max_iterations: int  # Safety cap (default: 3)
    output_dir: str      # Where temp images get saved

    