"""
Guardrail Agent: First node in the pipeline.

Validates the user's brief BEFORE any other agents run.
Checks for:
- Profanity / offensive language
- Self-harm / violence / dangerous content
- Sexually explicit content
- Content that violates advertising standards
- Off-topic requests (not related to billboard/ad generation)

If the brief fails validation, the pipeline short-circuits to END
with a rejection message. No images are generated, no API credits spent.
"""

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from state import BillboardState


GUARDRAIL_PROMPT = """You are a content safety and relevance validator for a billboard advertisement generation system.

Your job is to review a user's creative brief and determine if it is:
1. SAFE — no profanity, hate speech, self-harm, violence, sexually explicit content, or dangerous material
2. APPROPRIATE — suitable for public billboard advertising (visible to all ages including children)
3. RELEVANT — actually a request for a billboard or advertisement (not a random unrelated question)

Return ONLY valid JSON:
{
    "approved": true or false,
    "reason": "Brief explanation of why the brief was approved or rejected",
    "category": "safe" or "profanity" or "harmful" or "explicit" or "off_topic" or "ad_violation"
}

RULES:
- Billboards are public-facing — content must be appropriate for ALL audiences including children.
- Reject anything involving illegal products, weapons, hate groups, discriminatory messaging.
- Reject anything that is clearly not a billboard/ad request (e.g., "write me a poem", "what's the weather").
- Be reasonable — a brief like "beer advertisement" is fine (legal product), but "cigarette ad targeting teens" is not.
- When in doubt, approve — the planner agent will handle creative direction.
"""


def guardrail_agent(state: BillboardState) -> dict:
    """
    Guardrail node for LangGraph.
    Reads: state["brief"]
    Writes: state["guardrail_passed"], state["guardrail_message"]
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    messages = [
        SystemMessage(content=GUARDRAIL_PROMPT),
        HumanMessage(content=f"Creative brief to validate:\n\n{state['brief']}"),
    ]

    response = llm.invoke(messages)

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # If we can't parse the response, fail safe — block it
        result = {
            "approved": False,
            "reason": "Could not validate brief. Please try rephrasing.",
            "category": "safe",
        }

    return {
        "guardrail_passed": result.get("approved", False),
        "guardrail_message": result.get("reason", ""),
    }
