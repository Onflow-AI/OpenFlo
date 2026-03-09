# -*- coding: utf-8 -*-
"""
UX Synthesis Manager

Orchestrates SEQ scoring and SUS report generation for web automation sessions.
Follows the ChecklistManager pattern for consistency with the codebase architecture.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional, List

from openflo.ux.seq_scorer import SEQScorer
from openflo.ux.sus_calculator import SUSCalculator
from openflo.ux.report_generator import ReportGenerator
from openflo.prompts.ux_templates import (
    build_friction_analysis_prompt,
    parse_friction_response,
)


class UXSynthesisManager:
    """
    Manages UX evaluation by converting SEQ scores to SUS reports.

    This class orchestrates:
    1. SEQ score generation after each action
    2. SUS synthesis from accumulated SEQ data
    3. Friction point analysis
    4. Report generation (JSON and Markdown)

    Follows the ChecklistManager pattern for integration consistency.
    """

    def __init__(
        self,
        engine,
        logger: Optional[logging.Logger] = None,
        ux_engine=None,
        include_screenshots: bool = True,
        custom_seq_prompt: str = None,
    ):
        """
        Initialize UX Synthesis Manager.

        Args:
            engine: Main LLM engine for SUS synthesis
            logger: Optional logger instance
            ux_engine: Optional lightweight engine for SEQ scoring (defaults to main engine)
            include_screenshots: Whether to include screenshots in SEQ evaluation
            custom_seq_prompt: Optional custom SEQ prompt from config
        """
        self.engine = engine
        self.ux_engine = ux_engine or engine
        self.logger = logger or logging.getLogger("UXSynthesisManager")
        self.include_screenshots = include_screenshots
        self.custom_seq_prompt = custom_seq_prompt

        # Initialize components
        self.seq_scorer = SEQScorer(
            self.ux_engine, self.logger, custom_seq_prompt=custom_seq_prompt
        )
        self.sus_calculator = SUSCalculator(self.engine, self.logger)
        self.report_generator = ReportGenerator(self.logger)

        # Data storage
        self.seq_scores: List[dict] = []
        self.sus_report: Optional[dict] = None
        self.friction_points: List[dict] = []

        self.logger.info("UX Synthesis Manager initialized")

    async def evaluate_action_seq(
        self,
        action_data: dict,
        task_description: str,
        current_url: str,
        page_title: str,
        screenshot_path: Optional[str] = None,
    ) -> dict:
        """
        Evaluate SEQ score for a single action (called after each action).

        This method should be called after each action execution to record
        the perceived ease of that step.

        Args:
            action_data: Enhanced action record from taken_actions
            task_description: Current task being executed
            current_url: Current page URL
            page_title: Current page title
            screenshot_path: Optional screenshot for visual context

        Returns:
            SEQ evaluation result dict with score and thinking log
        """
        action_index = len(self.seq_scores)
        total_actions = action_index + 1

        # Only include screenshot if enabled
        screenshot = screenshot_path if self.include_screenshots else None

        try:
            seq_result = await self.seq_scorer.evaluate_action(
                action_data=action_data,
                task_description=task_description,
                page_url=current_url,
                page_title=page_title,
                action_index=action_index,
                total_actions=total_actions,
                screenshot_path=screenshot,
            )

            self.seq_scores.append(seq_result)

            self.logger.info(
                f"Multi-metric evaluation for step {seq_result['step']}: "
                f"SEQ={seq_result['seq_score']}/7, Eff={seq_result.get('efficiency', '?')}/7, "
                f"Clarity={seq_result.get('clarity', '?')}/7, Conf={seq_result.get('confidence', '?')}/7"
            )

            return seq_result

        except Exception as e:
            self.logger.error(f"SEQ evaluation failed for step {action_index + 1}: {e}")
            # Create fallback result with multi-metric structure (matching seq_scorer.py:141-152)
            fallback = {
                "step": action_index + 1,
                "seq_score": 4,  # Neutral
                "efficiency": 4,
                "efficiency_assessment": "Assessment unavailable due to evaluation error.",
                "clarity": 4,
                "clarity_assessment": "Assessment unavailable due to evaluation error.",
                "confidence": 4,
                "confidence_assessment": "Assessment unavailable due to evaluation error.",
                "thinking_log": f"Evaluation error: {str(e)}",
                "success": action_data.get("success", True),
            }
            self.seq_scores.append(fallback)
            return fallback

    async def generate_sus_report(
        self, task_description: str, task_id: str, output_path: str
    ) -> Optional[dict]:
        """
        Generate final SUS report from accumulated SEQ scores.

        This method should be called at session end (in agent.stop()).

        Args:
            task_description: The task that was executed
            task_id: Task identifier
            output_path: Directory to save reports

        Returns:
            Complete SUS report dict, or None if no data
        """
        if not self.seq_scores:
            self.logger.warning("No SEQ scores to synthesize - skipping SUS report")
            return None

        self.logger.info(
            f"Generating SUS report from {len(self.seq_scores)} SEQ scores..."
        )

        try:
            # Calculate averages for all metrics
            metric_averages = self._calculate_metric_averages()
            avg_seq = metric_averages["avg_seq"]

            # Aggregate qualitative assessments
            qualitative_summary = self._aggregate_qualitative_assessments()

            # Generate session summary
            session_summary = self._generate_session_summary()

            # Map multi-metrics to SUS using LLM
            self.logger.info("Mapping multi-metric scores to SUS framework...")
            sus_mapping = await self.sus_calculator.map_seq_to_sus(
                seq_data=self.seq_scores,
                task_description=task_description,
                session_summary=session_summary,
            )

            # Calculate final SUS score
            sus_result = self.sus_calculator.calculate_sus_score(
                sus_mapping.get("sus_items", [])
            )

            # Analyze friction points
            self.logger.info("Analyzing friction points...")
            friction_analysis = await self._analyze_friction_points()

            # Compile full report with all metrics
            self.sus_report = {
                "task_id": task_id,
                "task_description": task_description,
                "data_summary": {
                    "total_steps": len(self.seq_scores),
                    "average_seq_score": round(avg_seq, 2),
                    "average_efficiency": round(metric_averages["avg_efficiency"], 2),
                    "average_clarity": round(metric_averages["avg_clarity"], 2),
                    "average_confidence": round(metric_averages["avg_confidence"], 2),
                    "seq_scores": self.seq_scores,  # Now contains all 4 metrics per step
                    "qualitative_summary": qualitative_summary,
                },
                "sus_evaluation": {
                    "items": sus_mapping.get("sus_items", []),
                    "mapping_analysis": sus_mapping.get("mapping_analysis", ""),
                },
                "sus_calculation": sus_result,
                "friction_analysis": {
                    "key_friction_points": friction_analysis.get("friction_points", []),
                    "overall_assessment": friction_analysis.get(
                        "overall_assessment", ""
                    ),
                    "low_score_steps": [
                        s for s in self.seq_scores if s.get("seq_score", 7) <= 3
                    ],
                },
                "generated_at": datetime.now().isoformat(),
            }

            # Save reports
            self._save_reports(output_path)

            # Log summary
            summary = self.report_generator.generate_summary_text(self.sus_report)
            self.logger.info(f"SUS Report Complete: {summary}")

            return self.sus_report

        except Exception as e:
            self.logger.error(f"SUS report generation failed: {e}", exc_info=True)
            return None

    async def batch_evaluate_session(
        self, taken_actions: list, task_description: str, task_id: str, output_path: str
    ) -> Optional[dict]:
        """
        Batch evaluate an entire session's actions and generate SUS report.

        Useful for post-hoc analysis of completed sessions.

        Args:
            taken_actions: List of action records from a completed session
            task_description: The task that was executed
            task_id: Task identifier
            output_path: Directory to save reports

        Returns:
            Complete SUS report dict, or None if failed
        """
        self.logger.info(f"Batch evaluating {len(taken_actions)} actions...")

        # Reset state
        self.seq_scores = []
        self.sus_report = None

        # Evaluate all actions
        for i, action in enumerate(taken_actions):
            page_url = action.get("page_url", "Unknown")
            page_title = action.get("page_title", "Unknown")

            await self.evaluate_action_seq(
                action_data=action,
                task_description=task_description,
                current_url=page_url,
                page_title=page_title,
                screenshot_path=None,  # Screenshots typically not available in batch mode
            )

        # Generate report
        return await self.generate_sus_report(
            task_description=task_description, task_id=task_id, output_path=output_path
        )

    def _generate_session_summary(self) -> str:
        """Generate natural language summary of the session for LLM context."""
        if not self.seq_scores:
            return "No actions recorded."

        lines = []
        for seq in self.seq_scores:
            status = "Success" if seq.get("success", True) else "Failed"
            lines.append(
                f"Step {seq.get('step', '?')}: {seq.get('action_type', 'Unknown')} - "
                f"SEQ {seq.get('seq_score', '?')}/7 ({status})"
            )

        return "\n".join(lines)

    async def _analyze_friction_points(self) -> dict:
        """Identify and rank friction points from low SEQ scores."""
        low_scores = [s for s in self.seq_scores if s.get("seq_score", 7) <= 3]

        if not low_scores:
            return {
                "friction_points": [],
                "overall_assessment": "No significant friction points detected (all SEQ scores > 3).",
            }

        try:
            friction_prompt = build_friction_analysis_prompt(low_scores)
            response = await self.engine.generate(
                prompt=["", friction_prompt, ""],
                temperature=0.3,
                max_new_tokens=1000,
                turn_number=0,
            )

            # Extract response text
            if isinstance(response, list) and response:
                response_text = response[0] if response[0] else ""
            else:
                response_text = str(response) if response else ""

            # Parse response
            parsed = parse_friction_response(response_text)

            self.friction_points = parsed.get("friction_points", [])
            return parsed

        except Exception as e:
            self.logger.warning(f"Friction analysis failed: {e}")
            # Fallback: create basic friction point list from low scores
            friction_points = []
            for seq in low_scores:
                friction_points.append(
                    {
                        "step": seq.get("step", 0),
                        "severity": "medium"
                        if seq.get("seq_score", 4) >= 2
                        else "high",
                        "category": "unknown",
                        "description": seq.get(
                            "thinking_log", "Low SEQ score recorded"
                        ),
                        "recommendation": "Review step for usability issues",
                    }
                )

            return {
                "friction_points": friction_points,
                "overall_assessment": f"Analysis incomplete - {len(low_scores)} low-scoring steps detected.",
            }

    def _save_reports(self, output_path: str):
        """Save SUS report as JSON only."""
        if not self.sus_report:
            return

        # Ensure directory exists
        os.makedirs(output_path, exist_ok=True)

        # Save JSON
        try:
            json_path = os.path.join(output_path, "sus_report.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.sus_report, f, indent=2, ensure_ascii=False)
            self.logger.info(f"SUS report (JSON) saved to: {json_path}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON report: {e}")

    def get_current_seq_average(self) -> float:
        """Get current average SEQ score (useful for monitoring during execution)."""
        if not self.seq_scores:
            return 0.0
        return sum(s.get("seq_score", 4) for s in self.seq_scores) / len(
            self.seq_scores
        )

    def get_seq_score_count(self) -> int:
        """Get number of SEQ scores recorded."""
        return len(self.seq_scores)

    def _calculate_metric_averages(self) -> dict:
        """Calculate averages for all four metrics."""
        if not self.seq_scores:
            return {
                "avg_seq": 4.0,
                "avg_efficiency": 4.0,
                "avg_clarity": 4.0,
                "avg_confidence": 4.0,
            }

        return {
            "avg_seq": sum(s.get("seq_score", 4) for s in self.seq_scores)
            / len(self.seq_scores),
            "avg_efficiency": sum(s.get("efficiency", 4) for s in self.seq_scores)
            / len(self.seq_scores),
            "avg_clarity": sum(s.get("clarity", 4) for s in self.seq_scores)
            / len(self.seq_scores),
            "avg_confidence": sum(s.get("confidence", 4) for s in self.seq_scores)
            / len(self.seq_scores),
        }

    def _aggregate_qualitative_assessments(self) -> dict:
        """Extract representative qualitative assessments for reporting."""
        if not self.seq_scores:
            return {}

        # Get samples at different score ranges for each metric
        high_eff = [s for s in self.seq_scores if s.get("efficiency", 0) >= 6]
        low_eff = [s for s in self.seq_scores if s.get("efficiency", 0) <= 3]

        high_clarity = [s for s in self.seq_scores if s.get("clarity", 0) >= 6]
        low_clarity = [s for s in self.seq_scores if s.get("clarity", 0) <= 3]

        high_conf = [s for s in self.seq_scores if s.get("confidence", 0) >= 6]
        low_conf = [s for s in self.seq_scores if s.get("confidence", 0) <= 3]

        return {
            "efficiency_samples": {
                "high": {
                    "step": high_eff[0].get("step", "?") if high_eff else None,
                    "assessment": high_eff[0].get("efficiency_assessment", "")
                    if high_eff
                    else "",
                    "score": high_eff[0].get("efficiency", 0) if high_eff else None,
                },
                "low": {
                    "step": low_eff[0].get("step", "?") if low_eff else None,
                    "assessment": low_eff[0].get("efficiency_assessment", "")
                    if low_eff
                    else "",
                    "score": low_eff[0].get("efficiency", 0) if low_eff else None,
                },
            },
            "clarity_samples": {
                "high": {
                    "step": high_clarity[0].get("step", "?") if high_clarity else None,
                    "assessment": high_clarity[0].get("clarity_assessment", "")
                    if high_clarity
                    else "",
                    "score": high_clarity[0].get("clarity", 0)
                    if high_clarity
                    else None,
                },
                "low": {
                    "step": low_clarity[0].get("step", "?") if low_clarity else None,
                    "assessment": low_clarity[0].get("clarity_assessment", "")
                    if low_clarity
                    else "",
                    "score": low_clarity[0].get("clarity", 0) if low_clarity else None,
                },
            },
            "confidence_samples": {
                "high": {
                    "step": high_conf[0].get("step", "?") if high_conf else None,
                    "assessment": high_conf[0].get("confidence_assessment", "")
                    if high_conf
                    else "",
                    "score": high_conf[0].get("confidence", 0) if high_conf else None,
                },
                "low": {
                    "step": low_conf[0].get("step", "?") if low_conf else None,
                    "assessment": low_conf[0].get("confidence_assessment", "")
                    if low_conf
                    else "",
                    "score": low_conf[0].get("confidence", 0) if low_conf else None,
                },
            },
        }

    def reset(self):
        """Reset all stored data (useful for starting a new session)."""
        self.seq_scores = []
        self.sus_report = None
        self.friction_points = []
        self.logger.info("UX Synthesis Manager reset")
