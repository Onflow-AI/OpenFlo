# -*- coding: utf-8 -*-
"""
SEQ (Single Ease Question) Scorer

Evaluates individual actions on the SEQ scale (1-7) to measure
perceived ease of completing each step in a user flow.
"""

import logging
import os
from typing import Optional

from openflo.prompts.ux_templates import build_seq_evaluation_prompt, parse_seq_response


class SEQScorer:
    """
    Evaluates Single Ease Question scores for each action.

    SEQ Scale:
    1 = Very Difficult
    2 = Difficult
    3 = Somewhat Difficult
    4 = Neutral
    5 = Somewhat Easy
    6 = Easy
    7 = Very Easy
    """

    def __init__(
        self,
        engine,
        logger: Optional[logging.Logger] = None,
        custom_seq_prompt: str = None,
        persona=None,
    ):
        """
        Initialize SEQ Scorer.

        Args:
            engine: LLM engine for generating evaluations
            logger: Optional logger instance
            custom_seq_prompt: Optional custom SEQ prompt from config
            persona: Optional PersonaProfile to bias evaluation perspective
        """
        self.engine = engine
        self.logger = logger or logging.getLogger("SEQScorer")
        self.custom_seq_prompt = custom_seq_prompt
        self.persona = persona

    async def evaluate_action(
        self,
        action_data: dict,
        task_description: str,
        page_url: str,
        page_title: str,
        action_index: int = 0,
        total_actions: int = 1,
        screenshot_path: Optional[str] = None,
    ) -> dict:
        """
        Evaluate a single action and return SEQ score with thinking log.

        Args:
            action_data: Enhanced action record from taken_actions
            task_description: Current task being executed
            page_url: Current page URL
            page_title: Current page title
            action_index: 0-based index of this action
            total_actions: Total number of actions so far
            screenshot_path: Optional screenshot for visual context

        Returns:
            SEQ evaluation result dict:
            {
                "step": int,
                "seq_score": int (1-7),
                "efficiency": int (1-7),
                "efficiency_assessment": str,
                "clarity": int (1-7),
                "clarity_assessment": str,
                "confidence": int (1-7),
                "confidence_assessment": str,
                "thinking_log": str,
                "success": bool
            }
        """
        # Build page context
        page_context = {"url": page_url, "title": page_title}

        # Build prompt (use custom prompt if available)
        prompt = build_seq_evaluation_prompt(
            task_description=task_description,
            action_data=action_data,
            page_context=page_context,
            action_index=action_index,
            total_actions=total_actions,
            custom_prompt=self.custom_seq_prompt,
            persona=self.persona,
        )

        # Call LLM with optional screenshot
        try:
            if screenshot_path and os.path.exists(screenshot_path):
                response = await self.engine.generate(
                    prompt=["", prompt, ""],
                    image_path=screenshot_path,
                    temperature=0.3,
                    max_new_tokens=500,
                    turn_number=0,
                )
            else:
                response = await self.engine.generate(
                    prompt=["", prompt, ""],
                    temperature=0.3,
                    max_new_tokens=500,
                    turn_number=0,
                )

            # Extract response text
            if isinstance(response, list) and response:
                response_text = response[0] if response[0] else ""
            else:
                response_text = str(response) if response else ""

            # Parse response
            parsed = parse_seq_response(response_text)

        except Exception as e:
            self.logger.warning(f"SEQ evaluation LLM call failed: {e}")
            # Return default neutral scores on error
            parsed = {
                "seq_score": 4,
                "efficiency": 4,
                "efficiency_assessment": "Assessment unavailable due to evaluation error.",
                "clarity": 4,
                "clarity_assessment": "Assessment unavailable due to evaluation error.",
                "confidence": 4,
                "confidence_assessment": "Assessment unavailable due to evaluation error.",
                "thinking_log": f"Evaluation error: {str(e)}",
            }

        # Apply persona scoring bias if present
        bias = {}
        if self.persona and self.persona.scoring_bias:
            bias = self.persona.scoring_bias

        def _apply_bias(score, key):
            """Clamp score + bias offset to 1-7 range."""
            return max(1, min(7, score + bias.get(key, 0)))

        # Build complete result with multi-metric structure
        result = {
            "step": action_index + 1,
            "action_type": action_data.get("action_type", action_data.get("action", "UNKNOWN")),
            "seq_score": _apply_bias(parsed.get("seq_score", 4), "seq_modifier"),
            "efficiency": _apply_bias(parsed.get("efficiency", 4), "efficiency_modifier"),
            "efficiency_assessment": parsed.get("efficiency_assessment", ""),
            "clarity": _apply_bias(parsed.get("clarity", 4), "clarity_modifier"),
            "clarity_assessment": parsed.get("clarity_assessment", ""),
            "confidence": _apply_bias(parsed.get("confidence", 4), "confidence_modifier"),
            "confidence_assessment": parsed.get("confidence_assessment", ""),
            "thinking_log": parsed.get("thinking_log", ""),
            "success": action_data.get("success", True),
        }

        self.logger.debug(
            f"Multi-metric evaluation for step {result['step']}: "
            f"SEQ={result['seq_score']}/7, Eff={result['efficiency']}/7, "
            f"Clarity={result['clarity']}/7, Conf={result['confidence']}/7 - "
            f"({'Success' if result['success'] else 'Failed'})"
        )

        return result

    async def batch_evaluate(
        self, actions: list, task_description: str, page_contexts: Optional[list] = None
    ) -> list:
        """
        Evaluate multiple actions in batch (post-hoc analysis).

        Args:
            actions: List of action records
            task_description: Task being evaluated
            page_contexts: Optional list of page contexts per action

        Returns:
            List of SEQ evaluation results
        """
        results = []
        total_actions = len(actions)

        for i, action in enumerate(actions):
            # Get page context for this action
            if page_contexts and i < len(page_contexts):
                page_url = page_contexts[i].get("url", "Unknown")
                page_title = page_contexts[i].get("title", "Unknown")
            else:
                page_url = action.get("page_url", "Unknown")
                page_title = action.get("page_title", "Unknown")

            result = await self.evaluate_action(
                action_data=action,
                task_description=task_description,
                page_url=page_url,
                page_title=page_title,
                action_index=i,
                total_actions=total_actions,
            )
            results.append(result)

        return results
