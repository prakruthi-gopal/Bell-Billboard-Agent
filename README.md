# Billboard Ad Agent

An agentic system that generates and composes billboard advertisements for Bell Canada.

## Core idea

The billboard canvas is a **2D grid** (1024×300 pixels). Image generation models produce raw assets. The **Editor Agent** treats the canvas as a coordinate space — reasoning about where each asset goes, what size, how to crop — then uses image manipulation tools to compose the final billboard.

## Architecture

```
User Brief → Guardrail → Planner → Generator → Editor (ReAct loop) → Final Billboard
                |                                  ↑        |
           rejected → END                          |   fail + iterations remain
                                                   └────────┘
```

The pipeline is orchestrated in pure Python with a shared `BillboardState` dict passed between agents. Each agent reads what it needs from the state and writes its output back. The editor agent's ReAct loop uses a simple while condition — no framework abstraction hiding the logic. This keeps the agent behavior explicit and debuggable. In production, frameworks like LangGraph or CrewAI would add value for state persistence, parallel execution, and more complex multi-agent workflows.

**Guardrail Agent** — Validates the brief for safety, relevance, and advertising appropriateness. Blocks harmful or off-topic requests before any images are generated (no wasted API credits).

**Planner Agent** — Takes the creative brief and produces a structured spec: what image assets to generate (background, product, lifestyle), where each goes on the canvas (layout coordinates), headline text, brand overlay placement.

**Generator Agent** — Makes one Imagen API call per asset. A billboard with 3 visual elements = 3 Imagen calls. Each asset is saved separately with its role tagged.

**Editor Agent (the hero)** — Implements a ReAct loop (Reason → Act → Observe):
- **Reason**: LLM analyzes the assets, the planner's layout, and any compliance violations from previous iterations. Decides what tools to call with what parameters.
- **Act**: Composes the billboard — creates a fresh canvas, crops/resizes each asset to its target grid region, places them in layer order, adds brand overlay, headline text, and logo.
- **Observe**: Sends the composed billboard to Gemini Vision for brand compliance checking. If it fails, loops back with the violations as context.

Every iteration starts from a fresh canvas to prevent ghosting and double-layering.

## What the Editor Agent actually does (2D grid operations)

The key insight: image generation models are **tools**, not agents. The editor agent **orchestrates** them.

Each placement is a grid operation:
- `place_asset_on_canvas(x=512, y=40, width=380, height=220)` — maps an asset to a rectangular region on the canvas
- `crop_focus="center"` — decides which part of the source image survives the crop to match the target region's aspect ratio
- `resolve_overlap()` — rectangle intersection math to prevent assets from overlapping
- `validate_crop_ratio()` — catches extreme crops that would mangle faces or key content (e.g., cropping a portrait into a wide thin strip = 83% content loss)

These validations are **hardcoded in Python**, not LLM prompt suggestions. The LLM can output any coordinates it wants — the code enforces sanity.

## Tech stack

| Component | Technology |
|-----------|-----------|
| LLM reasoning | Gemini 2.5 Flash |
| Image generation | Google Imagen |
| Orchestration | Pure Python (shared state dict) |
| Image composition | Pillow |
| UI | Streamlit |

The model string for Imagen is a constant — swapping to Nano Banana means changing one line.

## Running locally

```bash
git clone <repo-url>
cd billboard_agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your API key to `.env`:
```
GEMINI_API_KEY=your_key_here
```

Run:
```
streamlit run app.py
```

## Production considerations (not built, but thought about)

- **Brand guidelines from config store** — currently hardcoded as a string constant. In production, load from a database or brand management system.
- **Agent frameworks (LangGraph, CrewAI, AutoGen)** — for state persistence, conditional routing, and multi-agent coordination at scale. For this POC, pure Python orchestration keeps the ReAct pattern explicit and the agent logic readable.
- **Persistent asset storage** — currently uses temp directories. Production would use S3/GCS.
- **Layout balance heuristics** — programmatic checks for visual weight distribution across the canvas.
- **A/B variant generation** — generate multiple billboard variants from one brief for testing.

## Project structure

```
billboard_agent/
├── app.py                  # Streamlit UI + pipeline orchestration
├── state.py                # Shared state schema
├── config.py               # Brand guidelines + constants
├── tools.py                # Pillow tools + hard validations
├── requirements.txt
├── .env                    # API key (gitignored)
└── agents/
    ├── guardrail.py        # Safety + relevance check
    ├── planner.py          # Brief → structured spec
    ├── generator.py        # Spec → Imagen assets
    └── editor.py           # ReAct composition loop
```
