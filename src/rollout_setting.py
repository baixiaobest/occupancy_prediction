from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class RollOutSetting:
    """Settings for generating rollouts from one or more scene templates.

    Fields:
        templates: List of scene-template objects.
        mirror: If True, data will be mirrored (not implemented yet).
        rotate: If True, data will be rotated (not implemented yet).
        name: Optional human-readable name for this rollout setting.
    """
    templates: List[Any] = field(default_factory=list)
    mirror: bool = False
    rotate: bool = False
    name: str | None = None
