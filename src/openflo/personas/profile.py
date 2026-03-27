# -*- coding: utf-8 -*-
"""
PersonaProfile dataclass for persona-based UX evaluation injection.
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class PersonaProfile:
    """
    Represents a human persona used to bias UX evaluation scoring.

    When injected into SEQ/SUS prompts, the LLM evaluates actions from
    this persona's perspective rather than as a generic neutral observer.
    """

    id: str
    display_name: str
    age_range: str
    digital_literacy: str        # "expert" | "intermediate" | "beginner" | "very_low"
    primary_device: str          # "desktop_keyboard" | "desktop_mouse" | "tablet_touch" | "mobile_touch"
    reading_speed: str           # "fast" | "normal" | "slow"
    tolerance_for_friction: str  # "high" | "medium" | "low" | "very_low"
    prior_experience: str        # Free text fed to the LLM
    common_friction_types: List[str] = field(default_factory=list)
    # Numeric post-parse offsets applied to each metric score after LLM response.
    # E.g. {"seq_modifier": -1, "clarity_modifier": -2} makes scores harsher.
    # Values are clamped to keep final scores in 1-7 range.
    scoring_bias: Dict[str, int] = field(default_factory=dict)
    description: str = ""        # 3-4 sentence narrative the LLM embodies

    def to_dict(self) -> dict:
        """Serialize to plain dict for JSON output."""
        return {
            "id": self.id,
            "display_name": self.display_name,
            "age_range": self.age_range,
            "digital_literacy": self.digital_literacy,
            "primary_device": self.primary_device,
            "reading_speed": self.reading_speed,
            "tolerance_for_friction": self.tolerance_for_friction,
            "prior_experience": self.prior_experience,
            "common_friction_types": self.common_friction_types,
            "scoring_bias": self.scoring_bias,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PersonaProfile":
        """
        Deserialize from a dict (e.g. from TOML inline table or task JSON).

        Raises:
            ValueError: If required fields are missing.
        """
        required = ["id", "display_name", "digital_literacy", "description"]
        missing = [f for f in required if not data.get(f)]
        if missing:
            raise ValueError(
                f"PersonaProfile missing required fields: {missing}. "
                f"Got keys: {list(data.keys())}"
            )

        return cls(
            id=data["id"],
            display_name=data["display_name"],
            age_range=data.get("age_range", "Unknown"),
            digital_literacy=data["digital_literacy"],
            primary_device=data.get("primary_device", "desktop_mouse"),
            reading_speed=data.get("reading_speed", "normal"),
            tolerance_for_friction=data.get("tolerance_for_friction", "medium"),
            prior_experience=data.get("prior_experience", ""),
            common_friction_types=data.get("common_friction_types", []),
            scoring_bias=data.get("scoring_bias", {}),
            description=data["description"],
        )
