# -*- coding: utf-8 -*-
# Copyright (c) 2024 OSU Natural Language Processing Group
#
# Licensed under the OpenRAIL-S License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.licenses.ai/ai-pubs-open-rails-vz1
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from seeact.prompts.templates import build_system_prompt, parse_tool_call, build_task_constraints_prompt, validate_parsed_action
from seeact.prompts.utils import llm_summarize_actions, llm_update_history_summary, analyze_repetitive_patterns
from urllib.parse import urlparse

async def predict(agent):
    """
    Generate a prediction for the next action using a unified tool-calling format.
    Single LLM call returns both reasoning and action in one response.
    Always returns a valid prediction dictionary, never None.
    """
    if not agent.initial_frame_saved:
        await agent.take_screenshot()
        agent.initial_frame_saved = True
    agent.time_step += 1
    
    # Check termination conditions
    if agent.valid_op >= agent.config['agent']['max_auto_op']:
        agent.logger.info(f"Reached maximum operations ({agent.config['agent']['max_auto_op']})")
        agent.complete_flag = True
        return {"action": "TERMINATE", "value": "Max operations reached", "element": None}
        
    if agent.continuous_no_op >= agent.config['agent']['max_continuous_no_op']:
        agent.logger.info(f"Reached maximum no-ops ({agent.config['agent']['max_continuous_no_op']})")
        agent.complete_flag = True
        return {"action": "TERMINATE", "value": "Max no-ops reached", "element": None}

    os.makedirs(os.path.join(agent.main_path, 'screenshots'), exist_ok=True)
    await agent.take_screenshot()
    if not agent.screenshot_path:
        agent.logger.error("Screenshot failed, terminating")
        agent.complete_flag = True
        return {"action": "TERMINATE", "value": "Screenshot failed", "element": None}

    try:
        if len(agent.taken_actions) > 0 and agent.time_step % agent.history_summary_interval == 0:
            older_end = max(0, len(agent.taken_actions) - agent.history_recent_window)
            if older_end > agent.llm_summary_covered_steps:
                delta = agent.taken_actions[agent.llm_summary_covered_steps:older_end]
                summary_engine = getattr(agent.checklist_manager, 'checklist_engine', agent.engine)
                if agent.llm_summary_covered_steps == 0 and older_end > 0 and not agent.llm_history_summary_text:
                    summary = await llm_summarize_actions(delta, summary_engine, logger=agent.logger)
                else:
                    summary = await llm_update_history_summary(delta, agent.llm_history_summary_text, summary_engine, logger=agent.logger)
                agent.llm_history_summary_text = summary
                agent.llm_summary_covered_steps = older_end
                if summary:
                    agent.logger.info(f"📝 History Summary: {summary}")
        
        # Format previous actions for context with success/failure information
        if agent.taken_actions:
            action_lines = []
            if agent.llm_history_summary_text:
                action_lines.append(f"📌 Summary: {agent.llm_history_summary_text}")
            start_idx = max(0, len(agent.taken_actions) - agent.history_recent_window)
            for i, action in enumerate(agent.taken_actions[-agent.history_recent_window:]):
                desc = action.get('action_description', 'Unknown action')
                success = action.get('success', True)
                error = action.get('error', None)
                if error:
                    status = f"❌ FAILED ({error})"
                elif success:
                    status = "✅ SUCCESS"
                else:
                    status = "⚠️ UNCERTAIN"
                action_lines.append(f"Step {start_idx + i + 1}: {desc} - {status}")
            previous_actions_text = "\n".join(action_lines)
            try:
                warnings_text = analyze_repetitive_patterns(agent.taken_actions)
                if warnings_text:
                    previous_actions_text += "\nWarnings: " + warnings_text
            except Exception:
                pass
        else:
            previous_actions_text = "No previous actions yet."
        
        # Get checklist context
        checklist_context = agent.checklist_manager.format_checklist_for_prompt() if agent.task_checklist else ""
        
        # Build unified prompt (generic tool-calling format)
        
        # Remove next-step hinting from prompts
        # Build task-specific soft constraints
        try:
            start_url = getattr(agent, 'actual_website', None) or agent.config.get("basic", {}).get("default_website")
            allowed_domain = urlparse(start_url).hostname or ""
        except Exception:
            allowed_domain = ""
        constraints_text = build_task_constraints_prompt(
            allowed_domain=allowed_domain,
            disallow_login=True,
            disallow_offsite=True,
            extra_rules=""
        )

        suggested_next = ""
        try:
            cs = agent.get_checklist_status()
            if cs.get("total", 0) > 0 and cs.get("completed", 0) >= cs.get("total", 0):
                suggested_next = "TERMINATE"
        except Exception:
            suggested_next = ""
        system_prompt, user_prompt = build_system_prompt(
            task=agent.tasks[-1],
            previous_actions=previous_actions_text,
            checklist_context=checklist_context,
            suggested_next_step=suggested_next,
            policy_constraints=constraints_text
        )
        
        agent.logger.info(f"Step - {agent.time_step}")
        agent.logger.info(f"TASK: {agent.tasks[-1]}")
        # agent.logger.info(f"Reasoning included: {bool(agent.task_reasoning)}") # Strategy removed from stepwise
        agent.logger.info("Previous actions:")
        for action in agent.taken_actions[-5:]:
            agent.logger.info(f"  - {action.get('action_description', 'Unknown')}")
        if agent.llm_history_summary_text:
            agent.logger.info(f"History Summary (current): {agent.llm_history_summary_text}")
        else:
            agent.logger.info("History Summary (current): None")
        if agent.task_checklist:
            snapshot = ", ".join([f"{item.get('id')}: {item.get('status','pending')}" for item in agent.task_checklist])
            agent.logger.info(f"Checklist snapshot: {snapshot}")
            full_ctx = agent.checklist_manager.format_checklist_for_prompt()
            if full_ctx:
                agent.logger.info(f"Checklist: {full_ctx}")
        
        parsed_action = None
        for attempt in range(3):
            repair = "" if attempt == 0 else "Your last output was invalid. Return EXACTLY one <tool_call> with name 'browser_use' and arguments containing 'action'. Include 'description' for all actions except ask_strategy. For action=ask_strategy include 'confusion'. Keep strings <=200 chars."
            # Guard image path existence to avoid FileNotFound errors
            screenshot_for_prediction = agent.screenshot_path if (agent.screenshot_path and os.path.exists(agent.screenshot_path)) else None
            action_model = agent.model
            try:
                if isinstance(action_model, str) and (":online" in action_model.lower()) and ("qwen" in action_model.lower()):
                    action_model = action_model.replace(":online", "")
            except Exception:
                pass
            user_with_repair = user_prompt if attempt == 0 else (user_prompt + "\n" + repair)
            llm_response = await agent.engine.generate(
                prompt=[system_prompt, user_with_repair, ""],
                image_path=screenshot_for_prediction,
                temperature=agent.temperature,
                model=action_model,
                turn_number=0
            )
            parsed_action = parse_tool_call(llm_response)
            agent.logger.info(f"Raw LLM response: {llm_response}")
            agent.logger.debug(f"Parsed action: {parsed_action}")
            if parsed_action and isinstance(parsed_action, dict):
                parsed_action.setdefault('value', '')
                parsed_action.setdefault('coordinates', None)
                parsed_action.setdefault('field', '')
                parsed_action.setdefault('action_description', '')
                if isinstance(parsed_action['value'], str) and len(parsed_action['value']) > 200:
                    parsed_action['value'] = parsed_action['value'][:200]
                if isinstance(parsed_action['field'], str) and len(parsed_action['field']) > 100:
                    parsed_action['field'] = parsed_action['field'][:100]
                if isinstance(parsed_action['action_description'], str) and len(parsed_action['action_description']) > 200:
                    parsed_action['action_description'] = parsed_action['action_description'][:200]
                coords = parsed_action.get('coordinates')
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    try:
                        parsed_action['coordinates'] = [int(coords[0]), int(coords[1])]
                    except Exception:
                        parsed_action['coordinates'] = None
                ok, reason = validate_parsed_action(parsed_action)
                if ok:
                    break
                else:
                    agent.logger.error(f"Schema invalid: {reason}")
                    parsed_action = None
        
        if not parsed_action or not parsed_action.get('action'):
            agent.logger.error("=" * 80)
            agent.logger.error("PARSING FAILURE - LLM Response Did Not Match Expected Format")
            agent.logger.error("=" * 80)
            agent.logger.error(f"Full LLM Response:\n{llm_response}")
            agent.logger.error("=" * 80)
            agent.logger.error("Expected format: <tool_call>{\"name\": \"browser_use\", \"arguments\": {action: ...}}</tool_call>")
            agent.logger.error("=" * 80)
            
            # Try to extract partial intent from malformed response
            llm_lower = llm_response.lower()
            if 'terminate' in llm_lower or 'complete' in llm_lower or 'done' in llm_lower:
                agent.logger.warning("Detected termination intent in malformed response, terminating...")
                return {"action": "TERMINATE", "value": "Task completion detected in malformed response", "element": None, "coordinates": None, "field": ""}
            elif 'scroll' in llm_lower:
                if 'down' in llm_lower:
                    agent.logger.warning("Detected scroll down intent in malformed response")
                    return {"action": "SCROLL DOWN", "value": "", "element": None, "coordinates": None, "field": ""}
                elif 'up' in llm_lower:
                    agent.logger.warning("Detected scroll up intent in malformed response")
                    return {"action": "SCROLL UP", "value": "", "element": None, "coordinates": None, "field": ""}
            
            # If can't extract anything useful, wait and continue (not NONE to avoid no-op counter)
            agent.logger.warning("Returning WAIT action to avoid counting malformed response as no-op")
            return {"action": "WAIT", "value": "1", "element": None, "coordinates": None, "field": ""}
        
        # Removed automatic ASK_STRATEGY trigger based on warnings/no-ops

        # Note: Action will be added to taken_actions in execute_action()
        # with full enhanced record including success/failure status
        
        return parsed_action
            
    except Exception as e:
        agent.logger.error(f"Prediction error: {e}", exc_info=True)
        agent.complete_flag = True
        return {"action": "TERMINATE", "value": f"Critical error: {e}", "element": None}
