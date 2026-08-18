# ==============================================================================
# untestable_ideas_menu.py -- Phase 5: The "Can't-Test-This-Yet" Menu
# ==============================================================================
# A browsable menu of strategy ideas the system CANNOT test yet because it is
# missing the specific data they need (order-book depth, funding rates, options
# surfaces, tick data, etc.). As the literature/LLM extraction finds ideas, the
# ones that need data you do not have are captured in the algorithm_ideas
# backlog instead of being silently dropped. This module is the VIEW over that
# backlog: it surfaces only the missing-data ideas, grouped by the data type
# they need, so you can scroll the names, pick the ones worth pursuing, and
# shelve the rest.
#
# WHAT IT IS AND IS NOT:
#   * It is a filtered, grouped, actionable VIEW over algorithm_ideas.IdeaBacklog.
#     It adds no new storage -- the backlog already records why_untestable and
#     data_needed. Building a second store would just create drift.
#   * It is NOT a tester. These ideas are, by definition, the ones the system
#     cannot evaluate. The menu's job is to keep them visible and organised so a
#     human can decide what to do, not to pretend they can be backtested.
#
# THE MISSING-DATA FILTER:
#   An idea belongs on this menu when it has a non-empty `data_needed` (it names
#   a data type it requires) AND it has not been resolved (status is open or
#   promising, not promoted or discarded). Ideas with no data_needed are
#   untestable for some OTHER reason and are handled elsewhere; this menu is
#   specifically the "we just don't have the data" list.
#
# DESIGN PRINCIPLE (project-wide):
#   The menu never implies an untestable idea is validated or good. Confidence
#   is shown as the human's own label ("speculative" / "promising" /
#   "ready-to-code"), never as a system judgement, and an idea with no evidence
#   is shown plainly as unevaluated rather than dressed up.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from algorithm_ideas import (
        IdeaBacklog, STATUS_OPEN, STATUS_PROMISING,
        STATUS_DISCARDED, STATUS_PROMOTED,
    )
    _HAVE_BACKLOG = True
except Exception:
    _HAVE_BACKLOG = False
    STATUS_OPEN, STATUS_PROMISING = "open", "promising"
    STATUS_DISCARDED, STATUS_PROMOTED = "discarded", "promoted"


# Statuses that still belong on the menu (unresolved).
_ACTIVE_STATUSES = {STATUS_OPEN, STATUS_PROMISING}


def _norm_data_type(data_needed: str) -> str:
    """
    Normalise a free-text data_needed string into a grouping key.

    People will write 'order book', 'order-book depth', 'L2 order book' etc.;
    this maps common phrasings to a canonical bucket so the menu groups them
    together instead of scattering near-duplicates. Unknown phrasings fall
    through to a cleaned version of the original text rather than a catch-all,
    so nothing is silently merged.
    """
    s = (data_needed or "").strip().lower()
    if not s:
        return "unspecified"
    buckets = {
        "order book": ("order book", "order-book", "orderbook", "l2", "depth of market", "dom"),
        "tick data": ("tick", "bid/ask tick", "quote data", "trade prints"),
        "funding rates": ("funding", "funding rate", "perp funding"),
        "open interest": ("open interest", "oi"),
        "options data": ("option", "options", "implied vol", "iv surface", "greeks"),
        "news / sentiment": ("news", "sentiment", "headline", "nlp"),
        "economic calendar": ("economic calendar", "econ calendar", "macro release", "nfp", "cpi surprise"),
        "on-chain": ("on-chain", "onchain", "wallet", "chain data"),
        "fundamentals": ("fundamental", "earnings", "balance sheet", "cot", "positioning"),
        "alternative data": ("satellite", "credit card", "web traffic", "alt data", "alternative data"),
    }
    for canonical, needles in buckets.items():
        if any(n in s for n in needles):
            return canonical
    # Unknown: keep a trimmed version of what the human wrote.
    return s[:40]


@dataclass
class MenuItem:
    idea_id: str
    title: str
    description: str
    data_needed: str
    data_type: str            # normalised bucket
    why_untestable: str
    category: str
    confidence: str
    status: str

    def one_line(self) -> str:
        conf = f" [{self.confidence}]" if self.confidence else ""
        return f"{self.title}{conf} - needs: {self.data_needed or 'unspecified'}"


@dataclass
class MenuGroup:
    data_type: str
    items: List[MenuItem] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.items)


class UntestableIdeasMenu:
    """Filtered, grouped view over the algorithm_ideas backlog."""

    def __init__(self, backlog: Optional[Any] = None):
        if backlog is not None:
            self.backlog = backlog
        elif _HAVE_BACKLOG:
            self.backlog = IdeaBacklog()
        else:
            self.backlog = None

    # -- Core query ------------------------------------------------------------
    def _all_active_missing_data(self) -> List[MenuItem]:
        if self.backlog is None:
            return []
        items: List[MenuItem] = []
        # Pull open + promising; exclude promoted/discarded.
        rows: List[Dict[str, Any]] = []
        for st in (STATUS_OPEN, STATUS_PROMISING):
            try:
                rows.extend(self.backlog.list_ideas(status=st))
            except Exception:
                continue
        for r in rows:
            data_needed = (r.get("data_needed") or "").strip()
            if not data_needed:
                continue  # untestable for a non-data reason; not this menu
            items.append(MenuItem(
                idea_id=r.get("idea_id", ""),
                title=r.get("title", "(untitled)"),
                description=r.get("description", ""),
                data_needed=data_needed,
                data_type=_norm_data_type(data_needed),
                why_untestable=r.get("why_untestable", ""),
                category=r.get("category", "uncategorized"),
                confidence=r.get("confidence", ""),
                status=r.get("status", STATUS_OPEN),
            ))
        return items

    def list_items(self, data_type: Optional[str] = None) -> List[MenuItem]:
        """Flat list of missing-data ideas, optionally filtered to one bucket."""
        items = self._all_active_missing_data()
        if data_type is not None:
            items = [i for i in items if i.data_type == data_type]
        # Stable, useful ordering: by data type, then title.
        items.sort(key=lambda i: (i.data_type, i.title.lower()))
        return items

    def grouped(self) -> List[MenuGroup]:
        """Ideas grouped by the data type they need, largest group first."""
        groups: Dict[str, MenuGroup] = {}
        for item in self._all_active_missing_data():
            g = groups.setdefault(item.data_type, MenuGroup(item.data_type))
            g.items.append(item)
        for g in groups.values():
            g.items.sort(key=lambda i: i.title.lower())
        return sorted(groups.values(), key=lambda g: (-g.count, g.data_type))

    def data_types(self) -> List[str]:
        """The distinct data types currently blocking ideas."""
        return sorted({i.data_type for i in self._all_active_missing_data()})

    # -- Actions (delegate to the backlog; no new state here) ------------------
    def mark_promising(self, idea_id: str) -> bool:
        """Flag an idea as worth pursuing when the data becomes available."""
        if self.backlog is None:
            return False
        return self.backlog.set_status(idea_id, STATUS_PROMISING)

    def discard(self, idea_id: str) -> bool:
        """Remove an idea from the menu (reviewed and rejected)."""
        if self.backlog is None:
            return False
        return self.backlog.set_status(idea_id, STATUS_DISCARDED)

    # -- Rendering (console; the Phase 6 UI can call the query methods) ---------
    def render(self) -> str:
        groups = self.grouped()
        if not groups:
            if self.backlog is None:
                return ("Untestable-ideas menu: backlog unavailable "
                        "(algorithm_ideas not importable).")
            return ("Untestable-ideas menu: nothing here. No open ideas are "
                    "currently blocked on missing data.")
        total = sum(g.count for g in groups)
        lines = [
            "=" * 64,
            " UNTESTABLE IDEAS -- blocked on data you don't have yet",
            "=" * 64,
            f" {total} idea(s) across {len(groups)} data type(s).",
            " These CANNOT be backtested here -- they are listed so you can",
            " pick ones to pursue manually or shelve until the data exists.",
            "",
        ]
        for g in groups:
            lines.append(f"-- {g.data_type.upper()}  ({g.count}) "
                         + "-" * max(0, 46 - len(g.data_type)))
            for it in g.items:
                lines.append(f"   * {it.one_line()}")
                if it.description:
                    lines.append(f"       {it.description[:80]}")
            lines.append("")
        lines.append("=" * 64)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Structured payload for the Phase 6 UI pane."""
        return {
            "total": len(self._all_active_missing_data()),
            "groups": [
                {
                    "data_type": g.data_type,
                    "count": g.count,
                    "items": [
                        {
                            "idea_id": it.idea_id,
                            "title": it.title,
                            "description": it.description,
                            "data_needed": it.data_needed,
                            "why_untestable": it.why_untestable,
                            "category": it.category,
                            "confidence": it.confidence,
                            "status": it.status,
                        }
                        for it in g.items
                    ],
                }
                for g in self.grouped()
            ],
        }


__all__ = ["UntestableIdeasMenu", "MenuItem", "MenuGroup"]


if __name__ == "__main__":
    menu = UntestableIdeasMenu()
    print(menu.render())
