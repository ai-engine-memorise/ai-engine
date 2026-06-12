"""Turn a PersonaExplanation into prose.

Default is a DETERMINISTIC template (no dependency, testable). An optional LLM
verbalizer is available for richer language; it consumes the SAME structured
explanation, so the facts are fixed and only the wording is generated.
"""
from __future__ import annotations
from typing import Optional

from ..contracts.models import PersonaExplanation

_STYLE = {
    "deep_reader": "reads stories closely, finishing what they start",
    "completionist": "works through many stories to completion",
    "skimmer": "skims widely, sampling many stories briefly",
    "sampler": "samples a range of stories at a moderate pace",
    "contemplative": "lingers on a few stories, reading slowly and reflectively",
    "unknown": "has not yet shown a clear reading style",
}
_PREF = {
    "object": "the objects and media themselves (photographs, artefacts)",
    "cognitive": "understanding the history and facts",
    "introspective": "personal reflection and individual testimony",
    "social": "stories that connect people",
    "unknown": "a still-forming set of interests",
}


def verbalize(exp: PersonaExplanation) -> str:
    """One-paragraph, evidence-grounded persona summary (deterministic)."""
    # truly cold only when there is no behavior at all (no interests, path, or aversions)
    if not exp.interests and not exp.trajectory and not exp.aversions:
        return ("New visitor — no engagement yet. Recommendations start from any survey "
                "persona or demographics, then adapt as they read.")

    parts: list[str] = []
    vt = exp.visitor_type
    if vt:
        parts.append(f"Looks like a **{vt.type}** ({vt.rationale}; confidence {vt.confidence:.2f}).")
    if exp.interests:
        parts.append(f"Drawn to {', '.join(i.label for i in exp.interests[:3])}.")
    if exp.experience_preference != "unknown":
        parts.append(f"Engages with a preference for {_PREF[exp.experience_preference]}; "
                     f"{_STYLE.get(exp.engagement_style, _STYLE['unknown'])}.")
    else:
        parts.append(f"So far {_STYLE.get(exp.engagement_style, _STYLE['unknown'])}.")
    if exp.aversions:
        parts.append(f"Tends to avoid {', '.join(a.label for a in exp.aversions[:2])}.")
    if exp.trajectory:
        parts.append("Recent path: " + " → ".join(exp.trajectory) + ".")
    return " ".join(parts)


def verbalize_llm(exp: PersonaExplanation, *, model: str = "claude-haiku-4-5-20251001") -> str:
    """LLM wording over the SAME structured facts (requires `anthropic` + API key).
    Falls back to the deterministic template if the SDK/key is unavailable."""
    try:
        from anthropic import Anthropic
    except ImportError:
        return verbalize(exp)
    facts = exp.model_dump(exclude={"summary"})
    prompt = (
        "You are describing a museum visitor to a curator, grounded ONLY in these "
        "structured facts (Falk visitor type, Pekarik experience preference, tag interests "
        "with evidence ids, engagement style). Write 2-3 sentences, warm and precise, invent "
        "nothing beyond the facts:\n\n" + str(facts)
    )
    try:
        client = Anthropic()
        msg = client.messages.create(model=model, max_tokens=300,
                                      messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip()
    except Exception:
        return verbalize(exp)
