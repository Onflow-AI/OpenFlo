# -*- coding: utf-8 -*-
"""
SUS (System Usability Scale) Calculator

Maps SEQ scores to SUS framework and calculates final usability scores.
"""

import logging
from typing import Optional

from avenir_ux.prompts.ux_templates import build_sus_synthesis_prompt, parse_sus_response


class SUSCalculator:
    """
    Maps SEQ scores to SUS framework and calculates final scores.

    SUS Items (1-5 scale: 1=Strongly Disagree, 5=Strongly Agree):
    - Odd items (1, 3, 5, 7, 9) are POSITIVE statements
    - Even items (2, 4, 6, 8, 10) are NEGATIVE statements

    SUS Score Calculation:
    - X = Sum of (Score - 1) for odd items (positive)
    - Y = Sum of (5 - Score) for even items (negative)
    - Final SUS Score = (X + Y) * 2.5 (range: 0-100)
    """

    SUS_ITEMS = [
        {
            "id": 1,
            "text": "I think that I would like to use this system frequently.",
            "positive": True,
        },
        {
            "id": 2,
            "text": "I found the system unnecessarily complex.",
            "positive": False,
        },
        {"id": 3, "text": "I thought the system was easy to use.", "positive": True},
        {
            "id": 4,
            "text": "I think that I would need the support of a technical person to be able to use this system.",
            "positive": False,
        },
        {
            "id": 5,
            "text": "I found the various functions in this system were well integrated.",
            "positive": True,
        },
        {
            "id": 6,
            "text": "I thought there was too much inconsistency in this system.",
            "positive": False,
        },
        {
            "id": 7,
            "text": "I would imagine that most people would learn to use this system very quickly.",
            "positive": True,
        },
        {
            "id": 8,
            "text": "I found the system very cumbersome to use.",
            "positive": False,
        },
        {"id": 9, "text": "I felt very confident using the system.", "positive": True},
        {
            "id": 10,
            "text": "I needed to learn a lot of things before I could get going with this system.",
            "positive": False,
        },
    ]

    # SUS grading scale based on Sauro-Lewis curved grading
    GRADE_SCALE = [
        (84.1, "A+", "Excellent", 96),
        (80.3, "A", "Excellent", 90),
        (78.8, "A-", "Good", 85),
        (77.1, "B+", "Good", 80),
        (74.1, "B", "Good", 70),
        (72.5, "B-", "Good", 65),
        (71.1, "C+", "OK", 60),
        (68.0, "C", "OK", 50),
        (51.0, "D", "Poor", 35),
        (0.0, "F", "Awful", 15),
    ]

    def __init__(self, engine, logger: Optional[logging.Logger] = None):
        """
        Initialize SUS Calculator.

        Args:
            engine: LLM engine for SUS mapping
            logger: Optional logger instance
        """
        self.engine = engine
        self.logger = logger or logging.getLogger("SUSCalculator")

    async def map_seq_to_sus(
        self,
        seq_data: list,
        task_description: str,
        session_summary: Optional[str] = None,
    ) -> dict:
        """
        Use LLM to map SEQ data to SUS item scores.

        Mapping Heuristics:
        - SEQ scores start low → high: High Learnability (Item 7)
        - SEQ scores volatile: High Inconsistency (Item 6)
        - SEQ consistently low for multi-step: High Complexity (Item 2)
        - Thinking logs mention "waiting/searching/retrying": High Cumbersomeness (Item 8)

        Args:
            seq_data: List of SEQ evaluation results
            task_description: The task being evaluated
            session_summary: Optional natural language summary

        Returns:
            {
                "sus_items": [{"id": 1, "score": 4, "rationale": "..."}, ...],
                "mapping_analysis": str
            }
        """
        if not seq_data:
            return self._get_default_sus_mapping("No SEQ data available")

        # Calculate average SEQ
        avg_seq = sum(s.get("seq_score", 4) for s in seq_data) / len(seq_data)

        # Generate session summary if not provided
        if not session_summary:
            session_summary = self._generate_session_summary(seq_data)

        # Build prompt
        prompt = build_sus_synthesis_prompt(
            task_description=task_description,
            seq_data=seq_data,
            session_summary=session_summary,
            average_seq=avg_seq,
        )

        try:
            response = await self.engine.generate(
                prompt=["", prompt, ""],
                temperature=0.3,
                max_new_tokens=2000,
                turn_number=0,
            )

            # Extract response text
            if isinstance(response, list) and response:
                response_text = response[0] if response[0] else ""
            else:
                response_text = str(response) if response else ""

            # Parse response
            parsed = parse_sus_response(response_text)

            # Validate we have all 10 items
            if len(parsed.get("sus_items", [])) != 10:
                self.logger.warning(
                    f"LLM returned {len(parsed.get('sus_items', []))} SUS items, expected 10. Using heuristic fallback."
                )
                return self._heuristic_sus_mapping(seq_data, task_description)

            return parsed

        except Exception as e:
            self.logger.warning(
                f"SUS mapping LLM call failed: {e}. Using heuristic fallback."
            )
            return self._heuristic_sus_mapping(seq_data, task_description)

    def calculate_sus_score(self, sus_items: list) -> dict:
        """
        Calculate final SUS score using standard formula.

        Formula:
        - X = Sum of (Score - 1) for odd items (1, 3, 5, 7, 9)
        - Y = Sum of (5 - Score) for even items (2, 4, 6, 8, 10)
        - Final SUS Score = (X + Y) * 2.5

        Args:
            sus_items: List of 10 SUS item dicts with 'id' and 'score'

        Returns:
            {
                "x_score": float (contribution from positive items),
                "y_score": float (contribution from negative items),
                "raw_sum": float (X + Y),
                "final_score": float (0-100),
                "grade": str (A+ to F),
                "adjective_rating": str (Excellent, Good, OK, Poor, Awful),
                "percentile": int (approximate percentile ranking)
            }
        """
        if not sus_items or len(sus_items) != 10:
            self.logger.warning(
                f"Invalid SUS items count: {len(sus_items) if sus_items else 0}"
            )
            return self._get_default_sus_result()

        # Sort items by ID to ensure correct order
        sorted_items = sorted(sus_items, key=lambda x: x.get("id", 0))

        x_score = 0  # Positive items (odd: 1, 3, 5, 7, 9)
        y_score = 0  # Negative items (even: 2, 4, 6, 8, 10)

        for item in sorted_items:
            item_id = item.get("id", 0)
            score = item.get("score", 3)

            # Clamp score to valid range
            score = max(1, min(5, score))

            if item_id % 2 == 1:  # Odd (positive)
                x_score += score - 1
            else:  # Even (negative)
                y_score += 5 - score

        raw_sum = x_score + y_score
        final_score = raw_sum * 2.5

        # Get grade, adjective, and percentile
        grade, adjective, percentile = self._get_sus_grade(final_score)

        return {
            "x_score": x_score,
            "y_score": y_score,
            "raw_sum": raw_sum,
            "final_score": round(final_score, 1),
            "grade": grade,
            "adjective_rating": adjective,
            "percentile": percentile,
        }

    def _get_sus_grade(self, score: float) -> tuple:
        """
        Convert SUS score to grade, adjective, and percentile.

        Based on Sauro-Lewis curved grading scale.

        Args:
            score: Final SUS score (0-100)

        Returns:
            Tuple of (grade, adjective, percentile)
        """
        for threshold, grade, adjective, percentile in self.GRADE_SCALE:
            if score >= threshold:
                return grade, adjective, percentile

        return "F", "Awful", 0

    def _generate_session_summary(self, seq_data: list) -> str:
        """Generate natural language summary of the session with all metrics."""
        if not seq_data:
            return "No actions recorded."

        lines = []
        for seq in seq_data:
            status = "Success" if seq.get("success", True) else "Failed"
            lines.append(
                f"Step {seq.get('step', '?')}: {seq.get('action_type', 'Unknown')} - "
                f"SEQ {seq.get('seq_score', '?')}/7, "
                f"Eff {seq.get('efficiency', '?')}/7, "
                f"Clarity {seq.get('clarity', '?')}/7, "
                f"Conf {seq.get('confidence', '?')}/7 ({status})"
            )

        return "\n".join(lines)

    def _heuristic_sus_mapping(self, seq_data: list, task_description: str) -> dict:
        """
        Fallback heuristic mapping when LLM call fails.

        Uses statistical analysis of all four metrics (SEQ, Efficiency, Clarity, Confidence) to estimate SUS items.
        """
        # Extract all metrics
        seq_scores = [s.get("seq_score", 4) for s in seq_data]
        efficiency_scores = [s.get("efficiency", 4) for s in seq_data]
        clarity_scores = [s.get("clarity", 4) for s in seq_data]
        confidence_scores = [s.get("confidence", 4) for s in seq_data]

        # Calculate averages
        avg_seq = sum(seq_scores) / len(seq_scores) if seq_scores else 4
        avg_efficiency = (
            sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 4
        )
        avg_clarity = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 4
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores) if confidence_scores else 4
        )

        # Map average SEQ (1-7) to SUS base score (1-5)
        # SEQ 1 -> SUS 1, SEQ 4 -> SUS 3, SEQ 7 -> SUS 5
        base_positive = round((avg_seq - 1) / 6 * 4 + 1)
        base_negative = 6 - base_positive  # Inverse for negative items

        # Check for volatility (inconsistency)
        if len(seq_scores) >= 3:
            variance = sum((s - avg_seq) ** 2 for s in seq_scores) / len(seq_scores)
            inconsistency_boost = 1 if variance > 2.5 else 0
        else:
            inconsistency_boost = 0

        # Check for learning curve (SEQ improvement)
        if len(seq_scores) >= 4:
            first_half = seq_scores[: len(seq_scores) // 2]
            second_half = seq_scores[len(seq_scores) // 2 :]
            learnability_boost = (
                1
                if (
                    sum(second_half) / len(second_half)
                    > sum(first_half) / len(first_half) + 0.5
                )
                else 0
            )
        else:
            learnability_boost = 0

        # Check for efficiency improvement (alternative learnability indicator)
        if len(efficiency_scores) >= 4:
            eff_first = efficiency_scores[: len(efficiency_scores) // 2]
            eff_second = efficiency_scores[len(efficiency_scores) // 2 :]
            if (
                sum(eff_second) / len(eff_second)
                > sum(eff_first) / len(eff_first) + 0.5
            ):
                learnability_boost = max(learnability_boost, 1)

        # Build SUS items with multi-metric adjustments
        sus_items = []
        for item in self.SUS_ITEMS:
            if item["positive"]:
                score = base_positive
                # Item 7 (Learnability) - modified by SEQ trends and efficiency improvements
                if item["id"] == 7:
                    score = min(
                        5, round((avg_efficiency - 1) / 6 * 4 + 1) + learnability_boost
                    )
                # Item 9 (Confidence) - directly from confidence metric
                elif item["id"] == 9:
                    score = min(5, round((avg_confidence - 1) / 6 * 4 + 1))
                # Item 3 (Ease of use) - modified by clarity
                elif item["id"] == 3:
                    score = min(5, round((avg_clarity - 1) / 6 * 4 + 1))
                # Item 1 (Frequency of use) - modified by confidence
                elif item["id"] == 1:
                    confidence_factor = round((avg_confidence - 1) / 6 * 4 + 1)
                    score = min(5, max(1, (base_positive + confidence_factor) // 2))
            else:
                score = base_negative
                # Item 6 (Inconsistency) - from SEQ volatility
                if item["id"] == 6:
                    score = min(5, score + inconsistency_boost)
                # Item 8 (Cumbersome) - inverse of efficiency
                elif item["id"] == 8:
                    score = min(5, round((8 - avg_efficiency) / 6 * 4 + 1))
                # Item 2 (Complex) - inverse of clarity
                elif item["id"] == 2:
                    score = min(5, round((8 - avg_clarity) / 6 * 4 + 1))
                # Item 4 (Need support) - inverse of clarity
                elif item["id"] == 4:
                    score = min(5, round((8 - avg_clarity) / 6 * 4 + 1))

            sus_items.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "score": max(1, min(5, score)),
                    "rationale": f"Heuristic: SEQ={avg_seq:.1f}/7, Eff={avg_efficiency:.1f}/7, Clarity={avg_clarity:.1f}/7, Conf={avg_confidence:.1f}/7",
                }
            )

        return {
            "sus_items": sus_items,
            "mapping_analysis": f"Heuristic mapping (LLM unavailable). "
            f"Avg SEQ: {avg_seq:.1f}/7, "
            f"Avg Efficiency: {avg_efficiency:.1f}/7, "
            f"Avg Clarity: {avg_clarity:.1f}/7, "
            f"Avg Confidence: {avg_confidence:.1f}/7, "
            f"Volatility: {'High' if inconsistency_boost else 'Normal'}, "
            f"Learning curve: {'Detected' if learnability_boost else 'None'}",
        }

    def _get_default_sus_mapping(self, reason: str) -> dict:
        """Return default neutral SUS mapping."""
        sus_items = [
            {"id": item["id"], "text": item["text"], "score": 3, "rationale": reason}
            for item in self.SUS_ITEMS
        ]
        return {"sus_items": sus_items, "mapping_analysis": reason}

    def _get_default_sus_result(self) -> dict:
        """Return default SUS calculation result."""
        return {
            "x_score": 10,
            "y_score": 10,
            "raw_sum": 20,
            "final_score": 50.0,
            "grade": "D",
            "adjective_rating": "Poor",
            "percentile": 35,
        }
