# -*- coding: utf-8 -*-
"""
Report Generator for SUS Reports

Generates human-readable SUS reports in various formats (Markdown, JSON).
"""

import json
import logging
from datetime import datetime
from typing import Optional


class ReportGenerator:
    """Generates human-readable SUS reports in various formats."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize Report Generator.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger("ReportGenerator")

    def generate_markdown(self, sus_report: dict) -> str:
        """
        Generate comprehensive Markdown report.

        Sections:
        1. Executive Summary
        2. Data Summary (Average SEQ, score distribution)
        3. SUS Evaluation Table (10 items with scores)
        4. Final SUS Calculation
        5. Key Friction Points
        6. Recommendations

        Args:
            sus_report: Complete SUS report dict

        Returns:
            Markdown formatted string
        """
        if not sus_report:
            return "# UX Synthesis Report\n\nNo data available."

        lines = []

        # Header
        lines.append("# UX Synthesis Report: SEQ to SUS Analysis")
        lines.append("")
        lines.append(f"*Generated: {sus_report.get('generated_at', datetime.now().isoformat())}*")
        lines.append("")

        # Task Information
        lines.append("## Task Information")
        lines.append("")
        lines.append(f"**Task ID**: `{sus_report.get('task_id', 'N/A')}`")
        lines.append("")
        lines.append(f"**Task Description**: {sus_report.get('task_description', 'N/A')}")
        lines.append("")

        # Executive Summary
        sus_calc = sus_report.get('sus_calculation', {})
        data_summary = sus_report.get('data_summary', {})

        lines.append("## Executive Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Steps | {data_summary.get('total_steps', 'N/A')} |")
        lines.append(f"| Average SEQ Score | {data_summary.get('average_seq_score', 'N/A')}/7 |")
        lines.append(f"| Average Efficiency | {data_summary.get('average_efficiency', 'N/A')}/7 |")
        lines.append(f"| Average Clarity | {data_summary.get('average_clarity', 'N/A')}/7 |")
        lines.append(f"| Average Confidence | {data_summary.get('average_confidence', 'N/A')}/7 |")
        lines.append(f"| **Final SUS Score** | **{sus_calc.get('final_score', 'N/A')}** |")
        lines.append(f"| Grade | {sus_calc.get('grade', 'N/A')} |")
        lines.append(f"| Rating | {sus_calc.get('adjective_rating', 'N/A')} |")
        lines.append(f"| Percentile | {sus_calc.get('percentile', 'N/A')}% |")
        lines.append("")

        # SUS Score Interpretation
        lines.append(self._get_sus_interpretation(sus_calc.get('final_score', 50)))
        lines.append("")

        # Data Summary - Multi-Metric Scores
        lines.append("## Multi-Metric Scores by Step")
        lines.append("")
        lines.append("| Step | Action | SEQ | Efficiency | Clarity | Confidence | Status |")
        lines.append("|------|--------|-----|------------|---------|------------|--------|")

        seq_scores = data_summary.get('seq_scores', [])
        for seq in seq_scores:
            step = seq.get('step', '?')
            action = seq.get('action_type', 'Unknown')[:15]
            seq_score = seq.get('seq_score', '?')
            efficiency = seq.get('efficiency', '?')
            clarity = seq.get('clarity', '?')
            confidence = seq.get('confidence', '?')
            outcome = '✓' if seq.get('success', True) else '✗'

            # Color-code SEQ scores using emoji indicators
            seq_indicator = self._get_score_indicator(seq_score)
            lines.append(f"| {step} | {action} | {seq_indicator} {seq_score} | {efficiency} | {clarity} | {confidence} | {outcome} |")

        # Add average row
        lines.append(f"| **Avg** | | **{data_summary.get('average_seq_score', 'N/A')}** | **{data_summary.get('average_efficiency', 'N/A')}** | **{data_summary.get('average_clarity', 'N/A')}** | **{data_summary.get('average_confidence', 'N/A')}** | |")
        lines.append("")

        # Add metric analysis section
        lines.append("## Metric Analysis")
        lines.append("")
        avg_seq = data_summary.get('average_seq_score', 4)
        avg_eff = data_summary.get('average_efficiency', 4)
        avg_clar = data_summary.get('average_clarity', 4)
        avg_conf = data_summary.get('average_confidence', 4)

        lines.append(f"- **Average SEQ**: {avg_seq}/7 ({self._get_rating_label(avg_seq)})")
        lines.append(f"- **Average Efficiency**: {avg_eff}/7 ({self._get_rating_label(avg_eff)})")
        lines.append(f"- **Average Clarity**: {avg_clar}/7 ({self._get_rating_label(avg_clar)})")
        lines.append(f"- **Average Confidence**: {avg_conf}/7 ({self._get_rating_label(avg_conf)})")
        lines.append("")

        # Qualitative Insights Section
        qualitative_summary = data_summary.get('qualitative_summary', {})
        if qualitative_summary:
            lines.append("## Qualitative Insights")
            lines.append("")

            # Efficiency Assessments
            eff_samples = qualitative_summary.get('efficiency_samples', {})
            if eff_samples.get('high') or eff_samples.get('low'):
                lines.append("### Efficiency Assessments")
                lines.append("")

                high_eff = eff_samples.get('high', {})
                if high_eff.get('step') is not None:
                    lines.append(f"**High Efficiency (Score {high_eff.get('score', '?')}/7, Step {high_eff.get('step', '?')})**:")
                    lines.append(f"> {high_eff.get('assessment', 'N/A')}")
                    lines.append("")

                low_eff = eff_samples.get('low', {})
                if low_eff.get('step') is not None:
                    lines.append(f"**Low Efficiency (Score {low_eff.get('score', '?')}/7, Step {low_eff.get('step', '?')})**:")
                    lines.append(f"> {low_eff.get('assessment', 'N/A')}")
                    lines.append("")

            # Clarity Assessments
            clarity_samples = qualitative_summary.get('clarity_samples', {})
            if clarity_samples.get('high') or clarity_samples.get('low'):
                lines.append("### Clarity Assessments")
                lines.append("")

                high_clarity = clarity_samples.get('high', {})
                if high_clarity.get('step') is not None:
                    lines.append(f"**High Clarity (Score {high_clarity.get('score', '?')}/7, Step {high_clarity.get('step', '?')})**:")
                    lines.append(f"> {high_clarity.get('assessment', 'N/A')}")
                    lines.append("")

                low_clarity = clarity_samples.get('low', {})
                if low_clarity.get('step') is not None:
                    lines.append(f"**Low Clarity (Score {low_clarity.get('score', '?')}/7, Step {low_clarity.get('step', '?')})**:")
                    lines.append(f"> {low_clarity.get('assessment', 'N/A')}")
                    lines.append("")

            # Confidence Assessments
            conf_samples = qualitative_summary.get('confidence_samples', {})
            if conf_samples.get('high') or conf_samples.get('low'):
                lines.append("### Confidence Assessments")
                lines.append("")

                high_conf = conf_samples.get('high', {})
                if high_conf.get('step') is not None:
                    lines.append(f"**High Confidence (Score {high_conf.get('score', '?')}/7, Step {high_conf.get('step', '?')})**:")
                    lines.append(f"> {high_conf.get('assessment', 'N/A')}")
                    lines.append("")

                low_conf = conf_samples.get('low', {})
                if low_conf.get('step') is not None:
                    lines.append(f"**Low Confidence (Score {low_conf.get('score', '?')}/7, Step {low_conf.get('step', '?')})**:")
                    lines.append(f"> {low_conf.get('assessment', 'N/A')}")
                    lines.append("")

        # SUS Evaluation Table
        lines.append("## SUS Evaluation (10 Items)")
        lines.append("")
        lines.append("*Scale: 1 = Strongly Disagree, 5 = Strongly Agree*")
        lines.append("")
        lines.append("| # | Statement | Score | Rationale |")
        lines.append("|---|-----------|-------|-----------|")

        sus_items = sus_report.get('sus_evaluation', {}).get('items', [])
        for item in sorted(sus_items, key=lambda x: x.get('id', 0)):
            item_id = item.get('id', '?')
            text = item.get('text', 'N/A')[:60]
            if len(item.get('text', '')) > 60:
                text += "..."
            score = item.get('score', '?')
            rationale = item.get('rationale', '')[:40]
            if len(item.get('rationale', '')) > 40:
                rationale += "..."

            # Indicate positive/negative items
            polarity = "(+)" if item_id % 2 == 1 else "(-)"
            lines.append(f"| {item_id} {polarity} | {text} | {score}/5 | {rationale} |")

        lines.append("")

        # Mapping Analysis
        mapping_analysis = sus_report.get('sus_evaluation', {}).get('mapping_analysis', '')
        if mapping_analysis:
            lines.append("### Mapping Analysis")
            lines.append("")
            lines.append(mapping_analysis)
            lines.append("")

        # SUS Calculation Breakdown
        lines.append("## SUS Calculation")
        lines.append("")
        lines.append("```")
        lines.append(f"X (Positive items: 1,3,5,7,9) = Sum of (Score - 1) = {sus_calc.get('x_score', '?')}")
        lines.append(f"Y (Negative items: 2,4,6,8,10) = Sum of (5 - Score) = {sus_calc.get('y_score', '?')}")
        lines.append(f"Raw Sum (X + Y) = {sus_calc.get('raw_sum', '?')}")
        lines.append(f"Final SUS Score = {sus_calc.get('raw_sum', '?')} * 2.5 = {sus_calc.get('final_score', '?')}")
        lines.append("```")
        lines.append("")

        # Key Friction Points
        friction_analysis = sus_report.get('friction_analysis', {})
        friction_points = friction_analysis.get('key_friction_points', [])

        if friction_points:
            lines.append("## Key Friction Points")
            lines.append("")

            for i, fp in enumerate(friction_points, 1):
                severity = fp.get('severity', 'medium').upper()
                severity_emoji = {"HIGH": "\u274c", "MEDIUM": "\u26a0\ufe0f", "LOW": "\u2139\ufe0f"}.get(severity, "\u2022")

                lines.append(f"### {i}. Step {fp.get('step', '?')} - {fp.get('category', 'Unknown').title()}")
                lines.append("")
                lines.append(f"**Severity**: {severity_emoji} {severity}")
                lines.append("")
                lines.append(f"**Issue**: {fp.get('description', 'No description')}")
                lines.append("")
                lines.append(f"**Recommendation**: {fp.get('recommendation', 'No recommendation')}")
                lines.append("")

            # Overall assessment
            overall = friction_analysis.get('overall_assessment', '')
            if overall:
                lines.append("### Overall Assessment")
                lines.append("")
                lines.append(overall)
                lines.append("")

        # Low Score Steps Summary with Qualitative Details
        low_score_steps = friction_analysis.get('low_score_steps', [])
        if low_score_steps:
            lines.append("## Low-Scoring Steps (SEQ <= 3)")
            lines.append("")
            for step in low_score_steps:
                lines.append(f"### Step {step.get('step', '?')} - {step.get('action_type', 'Unknown')} (SEQ {step.get('seq_score', '?')}/7)")
                lines.append("")

                # Show all qualitative assessments for low-scoring steps
                eff_score = step.get('efficiency', '?')
                eff_assessment = step.get('efficiency_assessment', 'N/A')
                lines.append(f"- **Efficiency** ({eff_score}/7): {eff_assessment}")

                clarity_score = step.get('clarity', '?')
                clarity_assessment = step.get('clarity_assessment', 'N/A')
                lines.append(f"- **Clarity** ({clarity_score}/7): {clarity_assessment}")

                conf_score = step.get('confidence', '?')
                conf_assessment = step.get('confidence_assessment', 'N/A')
                lines.append(f"- **Confidence** ({conf_score}/7): {conf_assessment}")

                # Include thinking log if available
                thinking_log = step.get('thinking_log', '')
                if thinking_log:
                    thinking_preview = thinking_log[:120] + "..." if len(thinking_log) > 120 else thinking_log
                    lines.append(f"- **Thinking Log**: {thinking_preview}")

                lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by UX Synthesis Agent (SEQ to SUS)*")

        return "\n".join(lines)

    def generate_json(self, sus_report: dict, indent: int = 2) -> str:
        """
        Generate JSON formatted report.

        Args:
            sus_report: Complete SUS report dict
            indent: JSON indentation level

        Returns:
            JSON formatted string
        """
        return json.dumps(sus_report, indent=indent, ensure_ascii=False, default=str)

    def _get_score_indicator(self, score: int) -> str:
        """Get emoji indicator for SEQ score."""
        if not isinstance(score, (int, float)):
            return "\u2754"  # Question mark

        if score >= 6:
            return "\ud83d\udfe2"  # Green circle (easy)
        elif score >= 4:
            return "\ud83d\udfe1"  # Yellow circle (neutral)
        else:
            return "\ud83d\udd34"  # Red circle (difficult)

    def _get_rating_label(self, score: float) -> str:
        """Get rating label for metric score on 1-7 scale."""
        if not isinstance(score, (int, float)):
            return "Unknown"

        if score >= 6:
            return "Very Good"
        elif score >= 5:
            return "Good"
        elif score >= 4:
            return "Moderate"
        elif score >= 3:
            return "Below Average"
        else:
            return "Poor"

    def _get_sus_interpretation(self, score: float) -> str:
        """Get interpretation text for SUS score."""
        if not isinstance(score, (int, float)):
            return "> SUS score interpretation unavailable."

        if score >= 84.1:
            return ("> **Excellent Usability**: This score puts the system in the top 10% of usability. "
                   "Users find it exceptionally easy to use with minimal friction.")
        elif score >= 72.5:
            return ("> **Good Usability**: This is an above-average score indicating a generally positive user experience. "
                   "Some improvements may enhance the experience further.")
        elif score >= 68:
            return ("> **OK Usability**: This score is around the industry average. "
                   "The system is usable but has room for improvement.")
        elif score >= 51:
            return ("> **Poor Usability**: Below average score suggesting significant usability issues. "
                   "Users may struggle with key tasks and require improvements.")
        else:
            return ("> **Very Poor Usability**: This score indicates serious usability problems. "
                   "Major redesign or improvements are recommended.")

    def generate_summary_text(self, sus_report: dict) -> str:
        """
        Generate a brief text summary suitable for logging.

        Args:
            sus_report: Complete SUS report dict

        Returns:
            Brief summary string
        """
        sus_calc = sus_report.get('sus_calculation', {})
        data_summary = sus_report.get('data_summary', {})

        return (
            f"SUS Report: Score={sus_calc.get('final_score', 'N/A')} "
            f"(Grade: {sus_calc.get('grade', '?')}, {sus_calc.get('adjective_rating', '?')}) | "
            f"Steps: {data_summary.get('total_steps', 0)} | "
            f"Avg SEQ: {data_summary.get('average_seq_score', 'N/A')}/7"
        )
