"""
LangGraph graph definition for the Billboard Ad Agent system.

Graph flow:
    START → guardrail → (check) → planner → generator → editor → (compliance)
                           |                                ↑          |
                      rejected                              |   fail + iters < max
                           ↓                                └──────────┘
                          END                                     |
                                                           pass OR max
                                                                  ↓
                                                                 END
"""

import os
import tempfile
from langgraph.graph import StateGraph, END

from state import BillboardState
from agents.guardrail import guardrail_agent
from agents.planner import planner_agent
from agents.generator import generator_agent
from agents.editor import editor_agent


def should_proceed_after_guardrail(state: BillboardState) -> str:
    """
    Conditional edge after guardrail:
    If brief is approved → proceed to planner
    If rejected → go directly to END (no images generated, no credits spent)
    """
    if state.get("guardrail_passed"):
        return "planner"
    return "end"


def should_continue_editing(state: BillboardState) -> str:
    """
    Conditional edge after editor:
    If compliance passes or max iterations reached → END
    Otherwise → loop back to editor
    """
    if state["compliance_status"] == "pass":
        return "end"

    if state["iteration_count"] >= state["max_iterations"]:
        return "end"

    return "editor"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph graph."""
    graph = StateGraph(BillboardState)

    # Add nodes
    graph.add_node("guardrail", guardrail_agent)
    graph.add_node("planner", planner_agent)
    graph.add_node("generator", generator_agent)
    graph.add_node("editor", editor_agent)

    # Entry point: guardrail first
    graph.set_entry_point("guardrail")

    # Guardrail → (planner or END)
    graph.add_conditional_edges(
        "guardrail",
        should_proceed_after_guardrail,
        {
            "planner": "planner",
            "end": END,
        },
    )

    # Linear flow: planner → generator → editor
    graph.add_edge("planner", "generator")
    graph.add_edge("generator", "editor")

    # Editor loop
    graph.add_conditional_edges(
        "editor",
        should_continue_editing,
        {
            "editor": "editor",
            "end": END,
        },
    )

    return graph.compile()


def run_pipeline(brief: str, max_iterations: int = 3) -> BillboardState:
    """
    Run the full billboard generation pipeline.
    Main entry point — called by the Streamlit UI.
    """
    output_dir = tempfile.mkdtemp(prefix="billboard_")
    os.makedirs(output_dir, exist_ok=True)

    initial_state: BillboardState = {
        "brief": brief,
        "guardrail_passed": None,
        "guardrail_message": None,
        "spec": None,
        "image_assets": [],
        "current_image_path": None,
        "edit_history": [],
        "compliance_status": None,
        "compliance_violations": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "output_dir": output_dir,
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    return final_state