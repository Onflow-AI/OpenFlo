# -*- coding: utf-8 -*-
"""
Trajectory Logger

Records every DOM state transition as a structured log entry:
    (timestamp, state_hash_before, action_type, state_hash_after)

This provides the raw data for quantitative UX metric calculation.
"""

import json
import logging
import os
from typing import List, Optional, Set


class TrajectoryLogger:
    """
    Records DOM state transitions during agent execution.

    Each transition captures a before/after state hash pair alongside the
    action that caused the transition. The accumulated log feeds directly
    into UXMetricsEvaluator for metric calculation.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("TrajectoryLogger")
        self.transitions: List[dict] = []

    def log_transition(
        self,
        timestamp: str,
        state_before: str,
        action_type: str,
        state_after: str,
    ) -> None:
        """
        Append a transition record to the log.

        Args:
            timestamp: ISO-format timestamp of the transition
            state_before: DOM state hash before the action
            action_type: The action that caused the transition (e.g. CLICK, TYPE)
            state_after: DOM state hash after the action
        """
        entry = {
            "timestamp": timestamp,
            "state_before": state_before,
            "action_type": action_type,
            "state_after": state_after,
        }
        self.transitions.append(entry)
        self.logger.debug(
            f"Trajectory: {state_before[:8]}..  --[{action_type}]-->  {state_after[:8]}.."
        )

    def get_transitions(self) -> List[dict]:
        """Return the full ordered list of transition records."""
        return list(self.transitions)

    def get_unique_states(self) -> Set[str]:
        """
        Return the set of all distinct DOM state hashes observed.

        Excludes sentinel values (__no_page__, __hash_error__) so metrics
        are computed only over real page states.
        """
        from openflo.ux.dom_hasher import is_sentinel

        seen: Set[str] = set()
        for t in self.transitions:
            for key in ("state_before", "state_after"):
                h = t.get(key, "")
                if h and not is_sentinel(h):
                    seen.add(h)
        return seen

    def save(self, output_path: str) -> None:
        """
        Save the trajectory log as JSON.

        Args:
            output_path: Directory path where trajectory_log.json will be written.
        """
        os.makedirs(output_path, exist_ok=True)
        json_path = os.path.join(output_path, "trajectory_log.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.transitions, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Trajectory log saved to: {json_path}")
        except Exception as e:
            self.logger.error(f"Failed to save trajectory log: {e}")

    def reset(self) -> None:
        """Clear all recorded transitions."""
        self.transitions = []
        self.logger.debug("Trajectory logger reset")
