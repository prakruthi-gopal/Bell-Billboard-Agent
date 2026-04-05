"""
Generator Agent: Produces MULTIPLE image assets for the billboard.

For a holiday Bell billboard, this might generate:
- A winter cityscape background
- A family lifestyle photo
- A Bell device/product shot

Each asset is a separate Imagen call. The editor agent then
composes them together onto the billboard canvas.
"""

import os
from google import genai
from google.genai import types

from state import BillboardState, ImageAsset


# IMAGEN_MODEL = "imagen-4.0-fast-generate-001"
IMAGEN_MODEL = "imagen-4.0-generate-001"

def generator_agent(state: BillboardState) -> dict:
    """
    Generator node for LangGraph.
    Reads: state["spec"]["image_assets"]
    Writes: state["image_assets"]
 
    HARD RULE: Always generates exactly 2 assets (1 background + 1 foreground).
    """
    spec = state["spec"]
    output_dir = state["output_dir"]
    asset_specs = spec.get("image_assets", [])
 
    # ENFORCE max 2 assets — but allow 1 if planner chose single asset approach
    if len(asset_specs) == 0:
        # Planner gave nothing — add a default
        asset_specs = [{
            "role": "background",
            "prompt": "Diverse group of happy people using smartphones, modern setting, natural lighting, no text, 8k",
            "description": "People using phones, centered in frame",
        }]
 
    if len(asset_specs) > 2:
        # Too many — keep only background + first foreground
        background = [a for a in asset_specs if a["role"] == "background"]
        foreground = [a for a in asset_specs if a["role"] != "background"]
        asset_specs = background[:1] + foreground[:1]
 
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
 
    generated_assets: list[ImageAsset] = []
 
    for i, asset_spec in enumerate(asset_specs):
        prompt = asset_spec["prompt"]
 
        result = client.models.generate_images(
            model=IMAGEN_MODEL,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type="image/png",
                aspect_ratio="16:9",
            ),
        )
 
        save_path = os.path.join(output_dir, f"asset_{i}_{asset_spec['role']}.png")
        result.generated_images[0].image.save(save_path)
 
        generated_assets.append({
            "role": asset_spec["role"],
            "path": save_path,
            "description": asset_spec.get("description", ""),
        })
 
    return {"image_assets": generated_assets}