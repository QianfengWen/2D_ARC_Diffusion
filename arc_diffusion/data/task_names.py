"""Human-friendly names for ARC synthetic tasks.

This file is intended to be user-editable. You can rename tasks here and the
rest of the tooling (dataset IDs, visualization filenames, metadata) will pick
up the changes automatically.
"""

import re
from typing import Dict, Optional


# Mapping from task code (used internally) to a friendly display name.
# Edit these names as you like. Keep them concise; filenames use a slugified
# version so spaces are okay but will be converted to dashes.
TASK_NAME_MAP: Dict[str, str] = {
    # Examples for the built-in synthetic tasks
    # code       -> display name
    "bb43febb": "fill red",
    "c9f8e694": "color row",
    "cbded52d": "two of three",
    "d4a91cb9": "L path",
    "d6ad076f": "narrow connect",
    # occ+color subset (original)
    "3aa6fb7a": "L corner fill",
    "1bfc4729": "two-tone frame",
    "0ca9ddb6": "color plus",
    # occ+color subset (new)
    "3e980e27": "template copy",
    "56ff96f3": "two-dot rectangle",
    "6c434453": "block to plus",
    "a699fb00": "row bridge",
    "e73095fd": "enclosed fill",
}


def slugify(name: str) -> str:
    """Turn a display name into a filesystem-friendly slug.

    Lowercase, replace non-alphanumerics with dashes, collapse repeats, and
    strip leading/trailing dashes.
    """
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-") or "task"


def get_task_display_name(code: str) -> str:
    """Friendly name for a task code. Falls back to the code itself."""
    return TASK_NAME_MAP.get(code, code)


def get_task_slug(code: str) -> str:
    """Slug for a task code derived from its display name."""
    return slugify(get_task_display_name(code))


def code_from_name_or_slug(value: str) -> Optional[str]:
    """Resolve a user-provided task identifier to a canonical code.

    Accepts any of:
      - exact code key present in TASK_NAME_MAP
      - display name value in TASK_NAME_MAP (case-insensitive)
      - slugified display name

    Returns the code if resolved, otherwise None.
    """
    if not value:
        return None
    v = value.strip()
    # Direct code
    if v in TASK_NAME_MAP:
        return v
    # By display name (case-insensitive)
    lower = v.casefold()
    for code, name in TASK_NAME_MAP.items():
        if name.casefold() == lower:
            return code
    # By slug
    for code, name in TASK_NAME_MAP.items():
        if slugify(name) == v or slugify(name) == lower:
            return code
    return None
