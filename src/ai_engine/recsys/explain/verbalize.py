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
    """One-paragraph summary led by the visitor's OWN behaviour data (how much they
    engaged, how deeply, what their interactions reveal they're drawn to), with the
    inferred Falk/Pekarik persona kept to a soft trailing clause. Deterministic."""
    # truly cold only when there is no behavior at all (no interests, path, or aversions)
    if not exp.interests and not exp.trajectory and not exp.aversions:
        return ("New visitor, no engagement yet. Recommendations start from any survey "
                "persona or demographics, then adapt as they read.")

    b = exp.behavior or {}
    n = int(b.get("n_views", 0) or 0)
    n_pos = int(b.get("n_positive", 0) or 0)
    n_neg = int(b.get("n_negative", 0) or 0)
    parts: list[str] = []

    # 1) activity — the raw counts
    if n:
        frag = f"Engaged with {n} {'story' if n == 1 else 'stories'}"
        bits = []
        if n_pos:
            bits.append(f"{n_pos} held attention")
        if n_neg:
            bits.append(f"{n_neg} dismissed")
        if bits:
            frag += ": " + ", ".join(bits)
        parts.append(frag + ".")

    # 2) reading depth — actual dwell / completion numbers
    dwell = b.get("avg_dwell_ratio")
    if isinstance(dwell, (int, float)) and n:
        comp = b.get("completion_rate") or 0.0
        parts.append(f"Reads to about {round(dwell * 100)}% of the estimated time, "
                     f"finishing {round(comp * 100)}%.")

    # 3) interests revealed by interactions, with the evidence count behind each
    if exp.interests:
        def _lab(i):
            ev = len(i.evidence or [])
            return i.label + (f" ({ev})" if ev else "")
        parts.append("Strongest pull toward " + ", ".join(_lab(i) for i in exp.interests[:3]) + ".")
    if exp.aversions:
        parts.append("Tends to avoid " + ", ".join(a.label for a in exp.aversions[:2]) + ".")
    if exp.trajectory:
        parts.append("Recent path: " + " → ".join(exp.trajectory) + ".")

    # 4) reading style + a soft persona tag (de-emphasized; no confidence jargon)
    style = _STYLE.get(exp.engagement_style)
    vt = exp.visitor_type
    art = "an" if (vt and vt.type and vt.type[:1].lower() in "aeiou") else "a"
    if style and vt and vt.type:
        parts.append(f"{style.capitalize()}; reads like {art} {vt.type.lower()}.")
    elif style:
        parts.append(style.capitalize() + ".")
    elif vt and vt.type:
        parts.append(f"Reads like {art} {vt.type.lower()}.")
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
