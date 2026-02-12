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

import string
import re
import os
import logging
import litellm
import asyncio
from seeact.llm.engine import add_llm_io_record
from seeact.prompts.templates import build_summary_prompt, build_summary_update_prompt

def analyze_repetitive_patterns(previous_actions):
    """Analyze previous actions for repetitive patterns and provide warnings"""
    if not previous_actions or len(previous_actions) < 2:
        return None
        
    # Get last 10 actions for analysis
    recent_actions = previous_actions[-10:] if len(previous_actions) > 10 else previous_actions
    
    # Enhanced failure detection - check multiple failure indicators
    def is_action_failed(action):
        if not isinstance(action, dict):
            return False
        
        # Check explicit failure indicators
        if action.get('success') is False or action.get('error'):
            return True
            
        # Check description for failure patterns (primary failure detection)
        desc = action.get('description', '').lower()
        failure_patterns = [
            'failed to', 'error', 'unsuccessful', 'could not', 'unable to',
            'did a click instead', 'no suitable option found', 'action failed'
        ]
        if any(pattern in desc for pattern in failure_patterns):
            return True
            
        # Special detection for SELECT actions that fallback to CLICK
        if (action.get('action', '').upper() == 'SELECT' and 
            ('did a click instead' in desc or 'no suitable option found' in desc)):
            return True
            
        return False
    
    # Track detailed action patterns
    action_patterns = []  # List of (action_type, element, value, is_failed)
    consecutive_failures = 0
    failed_actions = 0
    
    for action in recent_actions:
        if isinstance(action, dict):
            action_type = action.get('action', '').upper()
            element = action.get('element', '')
            value = action.get('value', '')
            is_failed = is_action_failed(action)
            
            action_patterns.append((action_type, element, value, is_failed))
            
            if is_failed:
                failed_actions += 1
                consecutive_failures += 1
            else:
                consecutive_failures = 0
    
    warnings = []
    
    # 1. Check for identical action repetitions (same action + element + value)
    if len(action_patterns) >= 3:
        for i in range(len(action_patterns) - 2):
            current = action_patterns[i]
            next_1 = action_patterns[i + 1] 
            next_2 = action_patterns[i + 2]
            
            # Check if 3 consecutive actions are identical and failed
            if (current[0] == next_1[0] == next_2[0] and  # same action type
                current[1] == next_1[1] == next_2[1] and  # same element
                current[2] == next_1[2] == next_2[2] and  # same value
                current[3] and next_1[3] and next_2[3]):  # all failed
                warnings.append(f"CRITICAL: You've attempted the identical action '{current[0]} {current[1]} {current[2]}' 3+ times and it keeps failing. STOP and try a completely different approach immediately!")
                break
    
    # 2. Enhanced SELECT action failure detection
    select_failures = []
    for i, (action_type, element, value, is_failed) in enumerate(action_patterns):
        if action_type == 'SELECT' and is_failed:
            select_failures.append((i, element, value))
    
    if len(select_failures) >= 2:
        # Check for repeated SELECT failures on same element with same value
        same_select_attempts = {}
        for _, element, value in select_failures:
            key = (element, value)
            same_select_attempts[key] = same_select_attempts.get(key, 0) + 1
        
        for (element, value), count in same_select_attempts.items():
            if count >= 2:
                warnings.append(f"SELECT action failing repeatedly on '{element}' with value '{value}' ({count} times). The dropdown may not contain this option or the element may not be a proper select. Try CLICKING to open dropdown first, or look for alternative elements.")
    
    # 3. Check for excessive consecutive failures (lowered threshold)
    if consecutive_failures >= 2:
        warnings.append(f"You have {consecutive_failures} consecutive failed actions. CHANGE STRATEGY NOW - try scrolling, different elements, or a completely different approach.")
    
    # 4. Check for high failure rate in recent actions
    if len(recent_actions) >= 3 and failed_actions / len(recent_actions) > 0.5:
        warnings.append(f"High failure rate: {failed_actions}/{len(recent_actions)} recent actions failed. Your current approach is not working - RECONSIDER YOUR STRATEGY completely.")
    
    # 5. Check for repetitive action types without progress
    action_types = [pattern[0] for pattern in action_patterns]
    if len(action_types) >= 4:
        last_4_actions = action_types[-4:]
        if len(set(last_4_actions)) == 1:  # All same action type
            failed_count = sum(1 for i in range(-4, 0) if action_patterns[i][3])
            if failed_count >= 3:
                warnings.append(f"You've been repeating the same failing action type '{last_4_actions[0]}' for 4 consecutive steps with {failed_count} failures. TRY A COMPLETELY DIFFERENT ACTION TYPE (scroll, wait, or terminate).")
    
    # 6. Early termination suggestion for obvious loops
    if len(recent_actions) >= 6:
        recent_failed = sum(1 for pattern in action_patterns[-6:] if pattern[3])
        if recent_failed >= 5:
            warnings.append("TERMINATION RECOMMENDED: You've failed 5+ times in the last 6 actions. This suggests the task may be impossible with current approach or the target element doesn't exist. Consider terminating or trying a fundamentally different strategy.")
    
    # 7. Special warning for SELECT -> CLICK fallback loops
    select_click_pattern = 0
    for i in range(len(action_patterns) - 1):
        current = action_patterns[i]
        next_action = action_patterns[i + 1]
        if (current[0] == 'SELECT' and current[3] and  # SELECT failed
            next_action[0] == 'CLICK' and current[1] == next_action[1]):  # followed by CLICK on same element
            select_click_pattern += 1
    
    if select_click_pattern >= 1:
        warnings.append("Detected SELECT->CLICK fallback pattern repeating. The element may not be a proper dropdown. Try a different element, use TYPE instead, or choose a different action type. Avoid repeating the same failing pattern.")
    
    # 8. Suggest changing approach for repeated element interaction failures
    element_failure_count = {}
    for pattern in action_patterns[-6:]:  # Check last 6 actions
        if pattern[3]:  # If action failed
            element = pattern[1]
            if element and element != 'none':
                element_failure_count[element] = element_failure_count.get(element, 0) + 1
    
    for element, count in element_failure_count.items():
        if count >= 2:
            warnings.append(f"Element '{element}' has failed {count} times in recent actions. Choose a different target or action type; avoid repeating the same failing interaction.")
    
    # 9. General suggestion for high failure rates
    if len(recent_actions) >= 3 and failed_actions >= 2:
        last_actions = [pattern[0] for pattern in action_patterns[-3:]]
        warnings.append("Multiple recent failures detected. Choose a fundamentally different approach: different action type, different element, or terminate if no viable path. Do not repeat the same action on the same element or coordinates.")
    
    return " ".join(warnings) if warnings else None

def generate_new_query_prompt(system_prompt="", task="", previous_actions=None, question_description="", repetition_detection=None):
    """
    Generate the first phase prompt to ask model to generate general descriptions about {environment, high-level plans, next step action}
    Each experiment will have a similar prompt in this phase
    This prompt is used to generate models' thoughts without disrupt of formatting/referring prompts
    
    Args:
        repetition_detection (dict): Result from repetitive action detection with forbidden patterns and suggestions
    """
    import logging
    # Use a more generic logger name that won't conflict with task-specific loggers
    logger = logging.getLogger(__name__)
    
    sys_role=""+system_prompt
    query_text = ""

    # Log prompt generation start
    logger.info("=== GENERATING QUERY PROMPT ===")
    logger.info(f"Task: {task}")
    logger.info(f"Previous actions count: {len(previous_actions) if previous_actions else 0}")

    # System Prompt
    query_text += "You are asked to complete the following task: "

    # Task Description
    query_text += task
    query_text += "\n\n"
    
    # Previous Actions with Enhanced Analysis
    previous_action_text = "Previous Actions:\n"
    if previous_actions is None:
        previous_actions = []
        logger.info("No previous actions to include")
    else:
        logger.info(f"Processing {len(previous_actions)} previous actions:")
        
    # Analyze for repetitive patterns
    repetitive_analysis = analyze_repetitive_patterns(previous_actions)
    if repetitive_analysis:
        previous_action_text += f"\n**REPETITIVE PATTERN ALERT**: {repetitive_analysis}\n\n"
    
    # Add repetition detection warnings if provided
    if repetition_detection and repetition_detection.get("has_repetition"):
        previous_action_text += "\n**CRITICAL: REPETITIVE ACTION DETECTED**\n"
        previous_action_text += "You have been repeating the same actions without success.\n"
        
        forbidden_patterns = repetition_detection.get("forbidden_patterns", [])
        if forbidden_patterns:
            previous_action_text += f"**FORBIDDEN ACTIONS**: You are PROHIBITED from using these patterns: {', '.join(forbidden_patterns)}\n"
        
        suggestions = repetition_detection.get("suggestions", [])
        if suggestions:
            previous_action_text += "**MANDATORY REQUIREMENTS**:\n"
            for suggestion in suggestions:
                previous_action_text += f"- {suggestion}\n"
        
        previous_action_text += "\n**YOU MUST CHOOSE A COMPLETELY DIFFERENT ACTION TYPE OR APPROACH**\n"
        previous_action_text += "If you cannot find a different approach, consider using TERMINATE.\n\n"
        
    for i, action_item in enumerate(previous_actions):
        if isinstance(action_item, dict):
            logger.info(f"  Action {i+1}: Step {action_item.get('step', 'N/A')} - {action_item.get('action_description', '')[:100]}...")
            step = action_item.get('step', 'N/A')
            action_type = action_item.get('predicted_action', action_item.get('action', 'UNKNOWN'))
            coords = action_item.get('coordinates')
            elem_center = action_item.get('element_center')
            desc = action_item.get('action_description', '')
            success = action_item.get('success', True)
            status = 'SUCCESS' if success else 'FAILED'
            previous_action_text += f"Step {step}: [{status}] {action_type}"
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                previous_action_text += f" at ({coords[0]},{coords[1]})"
            elif isinstance(elem_center, (list, tuple)) and len(elem_center) >= 2:
                previous_action_text += f" at ({elem_center[0]},{elem_center[1]})"
            if desc:
                previous_action_text += f" - {desc}"
            previous_action_text += "\n"
            if action_item.get('action_generation_response'):
                previous_action_text += f"  Action Generation: {action_item.get('action_generation_response', '')[:200]}...\n"
            if action_item.get('action_grounding_response'):
                previous_action_text += f"  Action Grounding: {action_item.get('action_grounding_response', '')[:200]}...\n"
            http_resp = action_item.get('http_response')
            if http_resp and isinstance(http_resp, dict):
                previous_action_text += f"  HTTP Response: {http_resp.get('status', 'N/A')} {http_resp.get('status_text', '')} - {http_resp.get('url', '')}\n"
            if action_item.get('error'):
                previous_action_text += f"  Error: {action_item.get('error', '')}\n"
        else:
            logger.info(f"  Action {i+1}: {str(action_item)[:100]}... (legacy format)")
            previous_action_text += str(action_item) + "\n"
    
    query_text += previous_action_text
    query_text += "\n"

    # Question Description
    query_text += question_description
    
    # Log final prompt structure
    logger.info(f"Generated system role length: {len(sys_role)} characters")
    logger.info(f"Generated query text length: {len(query_text)} characters")
    logger.info(f"Previous actions text length: {len(previous_action_text)} characters")
    logger.info("=== QUERY PROMPT GENERATION COMPLETE ===")
    
    return [sys_role,query_text]



def generate_new_referring_prompt(referring_description="", element_format="", action_format="", value_format="",
                              choices=None,split="4"):
    referring_prompt = ""

    # Add description about how to format output
    if referring_description != "":
        referring_prompt += referring_description
        referring_prompt += "\n\n"

    # Add element prediction format and choices


    # Prepare Option texts
    # For exp {1, 2, 4}, generate option
    # For element_atttribute, set options field at None
    if choices:
        choice_text = format_options(choices)
        referring_prompt += choice_text

    if element_format != "":
        referring_prompt += element_format
        referring_prompt += "\n\n"

    # Format Action Prediction
    if action_format != "":
        referring_prompt += action_format
        referring_prompt += "\n\n"

    # Format Value Prediction
    if value_format != "":
        referring_prompt += value_format
        referring_prompt += ""

    return referring_prompt

def format_options(choices):
    option_text = ""
    abcd = ''
    
    multi_choice = ''
    for multichoice_idx, choice in enumerate(choices):
        multi_choice += f"{generate_option_name(multichoice_idx)}. {choice}\n"
        abcd += f"{generate_option_name(multichoice_idx)}, "

    # Enhanced "none" option with clearer guidance
    multi_choice += f"none. **LAST RESORT ONLY** - Select this ONLY if: (1) Target element is absolutely not in the list AND (2) You've exhausted all alternatives (scrolling, similar elements, GUI_GROUNDING). **BEFORE selecting 'none'**: Try scrolling to find the element, consider similar elements that might work, or use GUI_GROUNDING if the element is visible but not numbered."
    # option_text += abcd
    option_text += f"**ELEMENT SELECTION PRIORITY**: First try to find a suitable element from the numbered options above. If your target element is not listed, consider: (1) Scrolling to find it, (2) Selecting a similar element that might achieve the same goal, (3) Using GUI_GROUNDING if the element is visible, (4) Only select 'none' as an absolute last resort.\n"
    option_text += (multi_choice + '\n\n')
    return option_text


def generate_option_name(index):
    if index < 26:
        return string.ascii_uppercase[index]
    else:
        first_letter_index = (index - 26) // 26
        second_letter_index = (index - 26) % 26
        first_letter = string.ascii_uppercase[first_letter_index]
        second_letter = string.ascii_uppercase[second_letter_index]
        return f"{first_letter}{second_letter}"

def get_index_from_option_name(name):
    # Handle "none" option specially
    if (name or "").lower() == "none":
        return -1
    
    if len(name or "") == 1:
        return string.ascii_uppercase.index((name or "")[0])
    elif len(name or "") == 2:
        first_letter_index = string.ascii_uppercase.index((name or "")[0])
        second_letter_index = string.ascii_uppercase.index((name or "")[1])
        return 26 + first_letter_index * 26 + second_letter_index
    else:
        # Log error instead of raising exception
        print(f"ERROR: Invalid string length for get_index_from_option_name: '{name}' (length: {len(name or '')})")
        print("The string should be either 1 or 2 characters long or 'none'")
        return None  # Return None instead of raising exception


def initialize_prompts():
    return {
        "system_prompt": "You are a web navigation assistant.",
        "action_space": "",
        "question_description": "What action should be performed next?",
        "referring_description": "Select the best element: {multichoice_question}",
        "element_format": "Element {id}: {description}",
        "action_format": "Action: {action}",
        "value_format": "Value: {value}"
    }


def generate_detailed_action_list(actions, start_idx=1) -> str:
    lines = []
    for i, action in enumerate(actions, start_idx):
        action_type = action.get('predicted_action', action.get('action', 'UNKNOWN'))
        desc = action.get('action_description', '')
        success = action.get('success', True)
        
        status = "✓" if success else "✗"
        lines.append(f"  Step {i} [{status}]: {action_type} - {desc}")
    
    return "\n".join(lines) if lines else "  No meaningful actions"


def generate_phase_summary(actions) -> str:
    if not actions:
        return "  No actions in this phase"
    
    action_counts = {}
    successful_actions = 0
    failed_actions = 0
    
    for action in actions:
        action_type = action.get('predicted_action', action.get('action', 'UNKNOWN'))
        action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        if action.get('success', True):
            successful_actions += 1
        else:
            failed_actions += 1
    
    action_summary = ", ".join([f"{count} {atype}" for atype, count in action_counts.items()])
    success_rate = f"{successful_actions}/{len(actions)} successful"
    
    key_actions = []
    if len(actions) > 0:
        first_desc = actions[0].get('action_description', 'Unknown')[:60]
        key_actions.append(f"Started: {first_desc}")
    
    if len(actions) > 1:
        last_desc = actions[-1].get('action_description', 'Unknown')[:60]
        key_actions.append(f"Ended: {last_desc}")
    
    return f"  Actions: {action_summary}\n  Success Rate: {success_rate}\n  Key Events:\n    - {chr(10).join(['    - ' + ka for ka in key_actions][1:]) if len(key_actions) > 1 else key_actions[0]}"


def generate_action_journey_summary(taken_actions) -> str:
    if not taken_actions:
        return "No actions taken."
    
    total_actions = len(taken_actions)
    
    if total_actions <= 10:
        return generate_detailed_action_list(taken_actions)
    
    summary_lines = []
    
    summary_lines.append("=== INITIAL PHASE (Steps 1-5) ===")
    summary_lines.append(generate_phase_summary(taken_actions[:5]))
    
    if total_actions > 15:
        middle_start = 5
        middle_end = total_actions - 5
        summary_lines.append(f"\n=== MIDDLE PHASE (Steps {middle_start+1}-{middle_end}) ===")
        summary_lines.append(generate_phase_summary(taken_actions[middle_start:middle_end]))
    
    summary_lines.append(f"\n=== FINAL PHASE (Steps {total_actions-4}-{total_actions}) ===")
    summary_lines.append(generate_detailed_action_list(taken_actions[-5:], start_idx=total_actions-4))
    
    return "\n".join(summary_lines)


def generate_action_summary(actions):
    if not actions:
        return ""
    
    summaries = []
    for action in actions:
        desc = action.get('action_description', '').strip()
        action_type = action.get('predicted_action', '')
        success = action.get('success', True)
        
        if not desc or desc.lower() in ['no action', 'unknown action', 'none']:
            continue
        
        if desc.startswith('Clicked at coordinates'):
            parts = desc.split('-')
            if len(parts) > 1:
                desc = parts[-1].strip()
            else:
                desc = "Clicked element"
        
        desc = re.sub(r'\(\d+,\s*\d+\)', '', desc).strip()
        
        if desc and desc[0].islower():
            desc = desc[0].upper() + desc[1:]
        
        if len(desc) > 3:
            if not success:
                desc = f"{desc} (failed)"
            summaries.append(desc)
    
    if not summaries:
        action_types = [a.get('predicted_action', '') for a in actions]
        clicks = action_types.count('CLICK')
        types = action_types.count('TYPE')
        if clicks > 0 and types > 0:
            return f"Clicked {clicks} elements and typed in {types} fields"
        elif clicks > 0:
            return f"Clicked {clicks} elements"
        elif types > 0:
            return f"Typed in {types} fields"
        return f"Performed {len(actions)} actions"
    
    if len(summaries) <= 3:
        return "; ".join(summaries)
    else:
        return f"{summaries[0]}; {summaries[1]}; ... {summaries[-1]}"


async def llm_summarize_actions(actions, engine, logger=None):
    try:
        if not actions:
            return ""
        lines = []
        for a in actions:
            if isinstance(a, dict):
                desc = a.get('action_description', '')
                success = a.get('success', True)
                status = "SUCCESS" if success else "FAILED"
                lines.append(f"- {desc} ({status})")
        summary_source = "\n".join(lines)[:2000]
        prompt = build_summary_prompt(summary_source)
        response = await engine.generate(prompt=["", prompt, ""], temperature=0.0, max_new_tokens=120, turn_number=0)
        if isinstance(response, list):
            return (response[0] or "").strip()
        if isinstance(response, str):
            return response.strip()
        if hasattr(response, 'choices') and response.choices:
            return (response.choices[0].message.content or "").strip()
        return ""
    except Exception:
        return generate_action_summary(actions)


async def llm_update_history_summary(delta_actions, previous_summary, engine, logger=None):
    try:
        if not delta_actions:
            return previous_summary
        prev = previous_summary or ""
        lines = []
        for a in delta_actions:
            if isinstance(a, dict):
                desc = a.get('action_description', '')
                success = a.get('success', True)
                status = "SUCCESS" if success else "FAILED"
                lines.append(f"- {desc} ({status})")
        delta_src = "\n".join(lines)[:2000]
        prompt = build_summary_update_prompt(prev, delta_src)
        response = await engine.generate(prompt=["", prompt, ""], temperature=0.0, max_new_tokens=140, turn_number=0)
        if isinstance(response, list):
            return (response[0] or "").strip()
        if isinstance(response, str):
            return response.strip()
        if hasattr(response, 'choices') and response.choices:
            return (response.choices[0].message.content or "").strip()
        return prev
    except Exception:
        addendum = await llm_summarize_actions(delta_actions, engine, logger=logger)
        if previous_summary:
            return f"{previous_summary} {addendum}".strip()
        return addendum
