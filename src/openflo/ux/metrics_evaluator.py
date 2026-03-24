# -*- coding: utf-8 -*-
"""
UX Metrics Evaluator

Computes three quantitative UX metrics from a DOM state transition log:

1. Trajectory Efficiency (E = S_reference / S_actual)
   Measures how directly the agent completed the task relative to a human
   reference trajectory. S_reference is the number of steps a human annotator
   took (from the benchmark dataset), not a theoretically computed minimum.

2. State Revisitation Rate (R = (S_actual - U_visited) / S_actual)
   Measures backtracking and confusion by counting repeated state visits.

3. Oscillation Count
   Counts A -> B -> A toggles, indicating high-friction loops where the agent
   cannot make progress and keeps bouncing between two states.
"""

import json
import logging
import os
from typing import List, Optional, Tuple


class UXMetricsEvaluator:
    """
    Calculates quantitative UX metrics from a trajectory log.

    Designed to complement the qualitative SEQ/SUS pipeline with
    deterministic, reproducible numbers.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("UXMetricsEvaluator")

    def evaluate(
        self,
        transitions: List[dict],
        reference_steps: Optional[int] = None,
    ) -> dict:
        """
        Calculate all UX metrics from the transition log.

        Args:
            transitions: List of transition dicts from TrajectoryLogger.
                Each dict has keys: timestamp, state_before, action_type, state_after.
            reference_steps: Optional number of steps in the human reference trajectory
                (e.g. Mind2Web's reference_length). When provided, trajectory_efficiency
                is computed as reference_steps / agent_steps. When absent, only the
                raw step count is reported.

        Returns:
            dict with all computed metrics and supporting data.
        """
        s_actual = len(transitions)

        efficiency = self._trajectory_efficiency(s_actual, reference_steps)
        revisitation_rate, unique_states, total_visits = self._state_revisitation(transitions)
        oscillation_count, oscillation_pairs = self._oscillation_count(transitions)

        metrics = {
            "agent_steps": s_actual,
            "reference_steps": reference_steps,
            "trajectory_efficiency": efficiency,
            "state_revisitation_rate": revisitation_rate,
            "unique_states_visited": unique_states,
            "total_state_visits": total_visits,
            "oscillation_count": oscillation_count,
            "oscillation_pairs": [list(pair) for pair in oscillation_pairs],
        }

        efficiency_str = f"{efficiency:.3f}" if efficiency is not None else "N/A (no reference)"
        self.logger.info(
            f"UX Metrics — Agent Steps: {s_actual}, "
            f"Efficiency: {efficiency_str}, "
            f"Revisitation: {revisitation_rate:.3f}, "
            f"Oscillations: {oscillation_count}"
        )
        return metrics

    def _trajectory_efficiency(
        self, s_actual: int, reference_steps: Optional[int]
    ) -> Optional[float]:
        """
        E = S_reference / S_actual, clamped to [0.0, 1.0].

        S_reference is the human-annotated reference trajectory length from the
        benchmark dataset. A value of 1.0 means the agent matched the human
        reference path length exactly. Values below 1.0 indicate the agent took
        more steps than the human reference.

        Returns None if no reference is available (cannot compute without a baseline).
        """
        if s_actual == 0:
            return None
        if not reference_steps or reference_steps <= 0:
            # No human reference available — cannot compute efficiency
            return None
        ratio = reference_steps / s_actual
        return round(min(1.0, max(0.0, ratio)), 4)

    def _state_revisitation(
        self, transitions: List[dict]
    ) -> Tuple[float, int, int]:
        """
        R_revisit = (S_actual - U_visited) / S_actual

        Where:
        - S_actual = total state visits (first state_before + all state_afters)
        - U_visited = number of unique state hashes (excluding sentinels)

        Returns:
            (revisitation_rate, unique_states_count, total_state_visits)
        """
        from openflo.ux.dom_hasher import is_sentinel

        if not transitions:
            return 0.0, 0, 0

        # Collect all visited states in order: first state_before, then all state_afters
        all_visits: List[str] = []
        first = transitions[0].get("state_before", "")
        if first and not is_sentinel(first):
            all_visits.append(first)

        for t in transitions:
            h = t.get("state_after", "")
            if h and not is_sentinel(h):
                all_visits.append(h)

        s_actual = len(all_visits)
        if s_actual == 0:
            return 0.0, 0, 0

        u_visited = len(set(all_visits))
        revisitation_rate = (s_actual - u_visited) / s_actual
        return round(revisitation_rate, 4), u_visited, s_actual

    def _oscillation_count(
        self, transitions: List[dict]
    ) -> Tuple[int, List[Tuple[str, str]]]:
        """
        Detect A -> B -> A oscillation patterns.

        An oscillation is detected when two consecutive transitions form a toggle:
        - transitions[i]:   state_before=A, state_after=B
        - transitions[i+1]: state_before=B, state_after=A
        with A != B (must be distinct states, not a no-op).

        Returns:
            (oscillation_count, list_of_oscillating_pairs)
        """
        from openflo.ux.dom_hasher import is_sentinel

        count = 0
        pairs: List[Tuple[str, str]] = []

        for i in range(len(transitions) - 1):
            t0 = transitions[i]
            t1 = transitions[i + 1]

            a0 = t0.get("state_before", "")
            b0 = t0.get("state_after", "")
            a1 = t1.get("state_before", "")
            b1 = t1.get("state_after", "")

            # Skip if any hash is a sentinel
            if any(is_sentinel(h) for h in [a0, b0, a1, b1]):
                continue

            # Check A -> B -> A: t0 goes A->B, t1 goes B->A (continuity + reversal)
            if a0 == b1 and b0 == a1 and a0 != b0:
                count += 1
                pairs.append((a0, b0))

        return count, pairs

    def save(self, metrics: dict, output_path: str) -> None:
        """
        Save the metrics report as JSON.

        Args:
            metrics: Output from evaluate()
            output_path: Directory path where step_count_trajectory_report.json will be written.
        """
        os.makedirs(output_path, exist_ok=True)
        json_path = os.path.join(output_path, "step_count_trajectory_report.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Step count trajectory report saved to: {json_path}")
        except Exception as e:
            self.logger.error(f"Failed to save UX metrics report: {e}")
