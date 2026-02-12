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

from .utils import generate_new_referring_prompt, generate_new_query_prompt


##### Specialized Prompt Builders for Agent Operations

def build_tools_definition() -> str:
    return """
<tools>
{
  "type": "function",
  "function": {
    "name": "browser_use",
    "description": "Single-step browser interaction using pixel coordinates or visible text.",
    "parameters": {
      "type": "object",
      "required": ["action"],
      "properties": {
        "action": {
          "type": "string",
          "enum": ["left_click", "hover", "keyboard", "type", "select", "press_enter", "scroll_up", "scroll_down", "scroll_top", "scroll_bottom", "new_tab", "close_tab", "go_back", "go_forward", "wait", "terminate"]
        },
        "coordinate": {"type": "array", "description": "Normalized [x,y] in 0–1000. REQUIRED for all actions except scroll; omit for scroll only. Include to target a container."},
        "text": {"type": "string", "description": "Visible label or input text. Use 'CLEAR' for keyboard.", "maxLength": 200},
        "code": {"type": "string", "description": "KeyboardEvent.code (e.g., 'PageDown', 'ArrowDown', 'Enter')", "maxLength": 50},
        "clear_first": {"type": "boolean", "description": "Clear active field before typing (type/keyboard)"},
        "press_enter_after": {"type": "boolean", "description": "Press Enter after typing (action=type)"},
        "field": {"type": "string", "description": "Semantic field name (email/search/password/country)", "maxLength": 100},
        "time": {"type": "number", "description": "Seconds to wait"},
        "status": {"type": "string", "enum": ["success", "failure"], "description": "Task status for terminate"},
        "description": {"type": "string", "description": "Short action description (<=200 chars). REQUIRED.", "maxLength": 200}
      }
    }
  }
}
</tools>

Screen: 1000×1000, origin (0,0) top-left.

Rules:
- Do not use GOTO for URL navigation.
- For <select> elements, YOU MUST use 'select' action directly. DO NOT use 'click' to open dropdowns.
- For all actions except scroll actions (scroll_up, scroll_down, scroll_top, scroll_bottom), YOU MUST provide the 'coordinate' parameter with normalized [x,y] values in 0–1000.
- keyboard: use 'code' for keys; 'text' for typing; 'CLEAR' clears the active field.
- **IMPORTANT**: You MUST provide 'coordinate' [x,y] for every CLICK, HOVER, or TYPE action. Do NOT rely on 'text' alone.

Return strictly in <tool_call> tags:
<tool_call>
{"name": "browser_use", "arguments": {"action": "...", ...}}
</tool_call>"""

def build_checklist_prompt(task_description: str, strategic_reasoning: str = "") -> str:
    """Concise checklist generation prompt with strict rules against hallucination."""
    strategy_block = ""
    if strategic_reasoning:
        strategy_block = f"\n\nStrategic Mindmap (Mental Model):\n{strategic_reasoning}\n\nUse this strategy to inform the checklist structure."

    return f"""Create 2–6 atomic outcome states based STRICTLY on the task description and the provided strategy.

Task: {task_description}{strategy_block}

Rules:
1) Each item is an observable goal state (not an action)
2) Max 10 words; short and specific
3) IDs: "requirement_1", "requirement_2", ...
4) Examples: "Size 'blue'", "T-shirt page", "Year: 2022-2023"
5) Status must be lowercase: pending, in_progress, completed, failed
6) DO NOT invent requirements not explicitly mentioned in the task.

Output JSON:
{{
    "checklist": [
        {{"id": "requirement_1", "description": "First outcome state", "status": "pending"}},
        {{"id": "requirement_2", "description": "Second outcome state", "status": "pending"}}
    ]
}}

Generate:"""


def build_checklist_update_prompt(action_type: str, success: bool, error: str, history_text: str,
                                  page_state_text: str, checklist_text: str) -> str:
    """Concise checklist update prompt with unchanged rules."""
    return f"""Update the checklist based on this action:

Action: {action_type} | Success: {success} | Error: {error if error else 'None'}

Recent actions:\n{history_text}
Page:\n{page_state_text[:300]}...
Checklist:\n{checklist_text}

Update rules:
• completed = fully satisfied
• in_progress = partially done
• pending = not started/reset
• failed = action failed
• Update exactly ONE item per action (most directly affected)
• new_status must be one of: pending, in_progress, completed, failed (lowercase)

Output JSON:
{{
    "updates": [
        {{"item_id": "requirement_X", "new_status": "pending", "reason": "Brief reason"}}
    ]
}}"""


def parse_tool_call(response_text: str) -> dict:
    """
    Parse Qwen3 tool-call response format (official browser_use format).
    
    Args:
        response_text: Raw response from LLM containing <tool_call> tags
    
    Returns:
        Dict with parsed action information or None if parsing fails
        
    Example input:
        <tool_call>
        {"name": "browser_use", "arguments": {"action": "left_click", "coordinate": [500, 300]}}
        </tool_call>
        
    Example output:
        {
            "action": "CLICK",
            "element": None,
            "value": None,
            "coordinates": [500, 300]
        }
    """
    import json
    import re
    
    try:
        matches = re.findall(r'<tool_call>(.*?)</tool_call>', response_text, re.DOTALL)
        if not matches:
            return None
        # Strictly require a single tool_call; reject if multiple present
        if len(matches) != 1:
            return None
        tool_call_json = matches[0].strip()
        # Strip code fences if present
        if tool_call_json.startswith('```'):
            tool_call_json = tool_call_json.split('\n', 1)[1] if '\n' in tool_call_json else tool_call_json
            if tool_call_json.endswith('```'):
                tool_call_json = tool_call_json[:-3]
        tool_call_json = tool_call_json.strip()
        # Attempt JSON parse; fallback to extracting JSON object substring
        try:
            tool_call_data = json.loads(tool_call_json)
        except Exception:
            obj_match = re.search(r'\{[\s\S]*\}', tool_call_json)
            if not obj_match:
                return None
            tool_call_data = json.loads(obj_match.group(0))
        
        # Validate structure
        if not isinstance(tool_call_data, dict) or 'name' not in tool_call_data or 'arguments' not in tool_call_data:
            return None
        
        name = tool_call_data['name']
        if name != 'browser_use':
            return None
        
        args = tool_call_data['arguments']
        action = args.get('action', '').lower()
        
        # Map browser_use action names to internal action names
        action_mapping = {
            'left_click': 'CLICK',
            'hover': 'HOVER',
            'drag': 'DRAG',
            'keyboard': 'KEYBOARD',
            'type': 'TYPE',
            'select': 'SELECT',
            'press_enter': 'PRESS ENTER',
            'scroll_up': 'SCROLL UP',
            'scroll_down': 'SCROLL DOWN',
            'scroll_top': 'SCROLL TOP',
            'scroll_bottom': 'SCROLL BOTTOM',
            'new_tab': 'NEW TAB',
            'close_tab': 'CLOSE TAB',
            'go_back': 'GO BACK',
            'go_forward': 'GO FORWARD',
            'goto': 'GOTO',
            'wait': 'WAIT',
            'terminate': 'TERMINATE'
        }
        
        mapped_action = action_mapping.get(action, action.upper())
        
        # Do not parse or include "next" suggestions; agent manages strategy separately
        
        # Build result dict
        coord = args.get('coordinate')
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            try:
                coord = [int(coord[0]), int(coord[1])]
            except Exception:
                coord = None
        else:
            coord = None
        
        if action == 'keyboard':
            # Prefer 'code' for key presses; fallback to 'text' for typing
            if isinstance(args.get('code', ''), str) and args.get('code', '').strip():
                val_text = args.get('code', '').strip()
            else:
                val_text = args.get('text', '') if isinstance(args.get('text', ''), str) else ''
        else:
            val_text = args.get('text', '') if isinstance(args.get('text', ''), str) else ''
        
        fld_text = args.get('field', '') if isinstance(args.get('field', ''), str) else ''
        desc_text = args.get('description', '') if isinstance(args.get('description', ''), str) else ''
        if len(val_text) > 200:
            val_text = val_text[:200]
        if len(fld_text) > 100:
            fld_text = fld_text[:100]
        if len(desc_text) > 200:
            desc_text = desc_text[:200]
        result = {
            'action': mapped_action,
            'element': None,
            'value': val_text,
            'coordinates': coord,
            'field': fld_text,
            'action_description': desc_text
        }
        if coord is not None:
            result['coordinates_type'] = 'normalized'
        # TYPE-specific optional flag to auto press Enter after typing
        if action == 'type':
            pe_after = args.get('press_enter_after', False)
            result['press_enter_after'] = bool(pe_after)
        if (not result['action_description'] or not result['action_description'].strip()):
            if mapped_action == 'CLICK':
                if coord:
                    result['action_description'] = f"Click at coordinates ({coord[0]}, {coord[1]})"
                else:
                    result['action_description'] = "Click element"
            elif mapped_action == 'HOVER':
                if coord:
                    result['action_description'] = f"Hover at coordinates ({coord[0]}, {coord[1]})"
                else:
                    result['action_description'] = "Hover element"
            elif mapped_action == 'TYPE':
                base = f"Type '{val_text}'" if val_text else "Type"
                result['action_description'] = base + (f" in {fld_text}" if fld_text else "")
            elif mapped_action == 'SELECT':
                base = f"Select '{val_text}'" if val_text else "Select"
                result['action_description'] = base + (f" from {fld_text}" if fld_text else "")
            elif mapped_action == 'PRESS ENTER':
                result['action_description'] = "Press Enter"
            elif mapped_action == 'SCROLL UP':
                result['action_description'] = "Scroll up"
            elif mapped_action == 'SCROLL DOWN':
                result['action_description'] = "Scroll down"
            elif mapped_action == 'SCROLL TOP':
                result['action_description'] = "Scroll to top"
            elif mapped_action == 'SCROLL BOTTOM':
                result['action_description'] = "Scroll to bottom"
            elif mapped_action == 'GOTO':
                result['action_description'] = f"Navigate to {val_text}" if val_text else "Navigate"
            elif mapped_action == 'KEYBOARD':
                if val_text and val_text.upper() == 'CLEAR':
                    result['action_description'] = "Clear field via keyboard"
                else:
                    base = f"Type '{val_text}' via keyboard" if val_text else "Type via keyboard"
                    result['action_description'] = base
            elif mapped_action == 'NEW TAB':
                result['action_description'] = "Open new tab"
            elif mapped_action == 'CLOSE TAB':
                result['action_description'] = "Close tab"
            elif mapped_action == 'GO BACK':
                result['action_description'] = "Go back"
            elif mapped_action == 'GO FORWARD':
                result['action_description'] = "Go forward"
        # Handle clear_first for TYPE/KEYBOARD with default true
        if mapped_action in ('TYPE', 'KEYBOARD'):
            cf = args.get('clear_first', True)
            if not isinstance(cf, bool):
                cf = True
            result['clear_first'] = cf
        
        # Handle special cases
        if action == 'drag':
            source_desc = args.get('source_desc') or args.get('source_text')
            target_desc = args.get('target_desc') or args.get('target_text')
            if source_desc or target_desc:
                result['drag_source_desc'] = source_desc
                result['drag_target_desc'] = target_desc
            else:
                target_coord = args.get('target_coordinate')
                if target_coord and len(target_coord) >= 2:
                    try:
                        result['drag_target_coords'] = [int(target_coord[0]), int(target_coord[1])]
                        result['drag_target_type'] = 'normalized'
                        result['value'] = f"{int(target_coord[0])},{int(target_coord[1])}"
                    except Exception:
                        result['value'] = ''
        elif action == 'wait':
            result['value'] = str(args.get('time', 1))
        elif action == 'terminate':
            result['value'] = ''
            result['status'] = args.get('status', 'success')
        elif action == 'goto':
            result['value'] = args.get('text', '')  # URL
        elif action == 'select':
            result['value'] = args.get('text', '')  # Option to select
        elif action == 'type':
            result['value'] = args.get('text', '')  # Text to type
        
        return result
        
    except Exception as e:
        return None


def build_task_constraints_prompt(allowed_domain: str = None,
                                  disallow_login: bool = True,
                                  disallow_offsite: bool = True,
                                  extra_rules: str = "") -> str:
    lines = ["Task-specific soft constraints:"]
    if disallow_login:
        lines += [
            "- Do NOT attempt to log in, sign in, sign up, or provide credentials.",
            "- If a login/sign-in UI is detected (password fields, 'Sign in', 'Log in', 'Create account'), TERMINATE immediately with status 'failure' and reason 'login prohibited'."
        ]
    return "\n".join(lines)


def validate_parsed_action(parsed_action: dict) -> (bool, str):
    if not isinstance(parsed_action, dict):
        return False, "not a dict"
    a = parsed_action.get('action')
    # next field removed from schema
    desc = parsed_action.get('action_description')
    if not a:
        return False, "missing action"
    # For other actions, desc is required.
    allowed = {"CLICK", "HOVER", "DRAG", "KEYBOARD", "TYPE", "SELECT", "PRESS ENTER", "SCROLL UP", "SCROLL DOWN", "SCROLL TOP", "SCROLL BOTTOM", "NEW TAB", "CLOSE TAB", "GO BACK", "GO FORWARD", "GOTO", "WAIT", "TERMINATE"}
    if a not in allowed:
        return False, "invalid action"
    val = parsed_action.get('value', '')
    fld = parsed_action.get('field', '')
    coords = parsed_action.get('coordinates', None)
    if isinstance(val, str) and len(val) > 200:
        return False, "value too long"
    if isinstance(fld, str) and len(fld) > 100:
        return False, "field too long"
    if isinstance(desc, str) and len(desc) > 200:
        return False, "description too long"
    if not isinstance(desc, str) or not desc.strip():
        return False, "missing description"
    if coords is not None:
        if not (isinstance(coords, (list, tuple)) and len(coords) >= 2 and all(isinstance(c, int) for c in coords[:2])):
            return False, "invalid coordinates"
    return True, ""

def build_system_prompt(task: str, previous_actions: str, checklist_context: str = "", suggested_next_step: str = "", policy_constraints: str = "") -> tuple:
    system_lines = [
        "One action per turn with pixel coordinates.",
        "CLICK: provide 'coordinate' or visible 'text'.",
        "Close/accept blocking modals, overlays, cookie banners first.",
        "Do not repeat actions unless page state visibly changed.",
        "TYPE/SELECT only when target field/dropdown is visible.",
        "KEYBOARD: use 'code' for keys; 'text' for typing; 'CLEAR' to clear active field.",
        "SCROLL: omit coordinates to scroll page; include [x,y] to scroll a container.",
        "If you see potential <select> elements, MUST use 'select' action directly. DO NOT use 'click' to open dropdowns.",
        "When objectives are achieved, TERMINATE with status 'success'.",
    ]
    tools_definition = build_tools_definition()
    system_text = "\n".join(system_lines) + "\n" + tools_definition

    user_lines = [
        "Task:",
        task,
        "Pre-step:",
        "Close or accept any cookie/consent banner before other actions.",
        "Constraints:",
        policy_constraints if policy_constraints else "No task-specific constraints.",
        "Previous actions:",
        previous_actions if previous_actions else "No previous actions yet.",
    ]
    if checklist_context:
        user_lines += ["", "Task progress:", checklist_context]
    if suggested_next_step:
        user_lines += ["", "Suggested next move:", f"{suggested_next_step} (suggested)"]
    user_text = "\n".join(user_lines)
    return system_text, user_text


def build_reasoning_prompt(task_description: str, website: str, policy_constraints: str = "") -> tuple:
    system_prompt = """You are a human UX evaluator performing a comprehensive cognitive walkthrough of a website.
Your goal is to simulate a human user's mindset, anticipating needs, identifying potential friction points, and structuring a mental model of how to achieve the task.

Create a comprehensive "mindmap" (structured as a hierarchical plan) that breaks down the task into:
1.  **User Intent & Mental Model**: What is the user trying to achieve? What are their expectations?
2.  **Strategic Approach**: High-level strategy to navigate the site.
3.  **Key Landmarks**: Expected navigation menus, buttons, or sections to look for.
4.  **Potential Pitfalls**: ambiguous labels, hidden menus, or confusing flows to avoid.

Return only:
<plan>...comprehensive structured mindmap...</plan>"""

    user_prompt = f"""Website: {website}
Task: {task_description}

Generate a comprehensive UX evaluation mindmap for this task.
- Mimic a human user's thought process.
- Be detailed and cover multiple possibilities.
- Do not just list actions; explain the *strategy* and *mental model*.

Return ONLY <plan>...</plan>; any other content will be ignored."""

    if policy_constraints and policy_constraints.strip():
        user_prompt = user_prompt + "\\n\\nConstraints:\\n" + policy_constraints.strip()
        
    return system_prompt, user_prompt


def format_reasoning_for_prompt(reasoning: str) -> str:
    """
    Format reasoning for inclusion in agent prompts.
    
    Args:
        reasoning: The reasoning text to format
        
    Returns:
        Formatted string for prompt inclusion
    """
    if not reasoning or not reasoning.strip():
        return ""
    
    return f"""
**STRATEGIC GUIDANCE FROM REASONING MODEL**:
{reasoning}

Use this guidance to inform your action selection, but adapt based on what you actually observe on the page.
"""


def build_termination_prompt(task: str, page_title: str, current_url: str, actions_count: int, action_summary: str) -> str:
    return f"""
You are an expert web automation strategist with advanced semantic reasoning capabilities. Analyze the current task execution state using hierarchical goal decomposition to make an intelligent termination decision.

**EXECUTION CONTEXT**
Original Task: {task}
Current Page: {page_title} ({current_url})
Actions Completed: {actions_count}

**EXECUTION HISTORY ANALYSIS**
{action_summary}

**CRITICAL BLOCKING CONDITIONS - IMMEDIATE TERMINATION**
Before any other analysis, check for these blocking conditions that require immediate termination:

1. **Access Denied/Forbidden**: "Access Denied", "403 Forbidden", "Permission denied", geographic restrictions
2. **Authentication Required**: "Sign in required", "Login required" (when login not part of task)
3. **Page Load Failures**: "Page not found", "Server error", persistent loading failures, blank pages
4. **Structural Blocking**: Page layout prevents task completion (missing essential elements, broken functionality)
5. **Contextual Blocking**: Current page context doesn't match task requirements (wrong website, irrelevant content)
6. **Workflow Blocking**: Page structure blocks intended workflow (modal dialogs, overlay screens, redirect loops)
7. **Progress Impossibility**: Essential functionality not available or accessible

**If ANY blocking condition is detected, immediately respond with TERMINATE.**

**ADVANCED SEMANTIC DECISION FRAMEWORK**
Only proceed with this analysis if NO blocking conditions are detected:

**SEMANTIC TASK DECOMPOSITION**
1. **Primary Goal Analysis**: What is the user's ultimate objective?
2. **Sub-goal Progress Mapping**: Which semantic sub-goals have been achieved?
3. **Completion Pathway Assessment**: What remains to fully satisfy user intent?
4. **Semantic Proximity Evaluation**: How close are we to meaningful task completion?

**HIERARCHICAL PROGRESS EVALUATION**

**Layer 1: Syntactic Progress**
- Have literal task actions been performed?
- Are we following the expected navigation path?

**Layer 2: Semantic Progress**
- Is the agent moving toward the user's underlying intent?
- Are we accessing the type of content/functionality the user seeks?
- Would a human recognize this as progress toward their goal?

**Layer 3: Contextual Progress**
- Is the current state enabling the user's likely next actions?
- Are we positioned to deliver the complete user experience they expect?
- Does the current page state satisfy the implicit requirements of the task?

**INTELLIGENT COMPLETION PROXIMITY ANALYSIS**
For "open/view/find" tasks:
- Is target content VISIBLE and ACCESSIBLE in current state?
- Can user immediately interact with or consume the target information?
- Are we on the correct page type with the right content focus?

For "navigate to" tasks:
- Are we on the specific page type mentioned in the task?
- Does the page contain the expected content/functionality?
- Is this where a human would expect to land for this task?

**ENHANCED BLOCKING PATTERN DETECTION**
Look for subtle blocking patterns that may not be immediately obvious:
- **Repetitive Failure Loops**: Same actions failing repeatedly without progress
- **Navigation Dead Ends**: Pages that don't lead toward task completion
- **Functional Limitations**: Missing capabilities needed for task completion
- **Content Accessibility Issues**: Target content exists but is not accessible
- **Workflow Interruptions**: Unexpected redirects or page changes that break task flow

**STRATEGIC CONTINUATION ASSESSMENT**

**STRONG BIAS TOWARD CONTINUATION**: The system should persist and try multiple approaches before giving up.

**High-Value Continuation Indicators** (PRIORITIZE THESE):
- ANY semantic progress toward user intent, even minimal
- Current approach showing ANY promise for goal achievement
- Failures are tactical (wrong elements) not strategic (wrong approach)
- Alternative pathways visible and viable
- User goal POTENTIALLY achievable from current state
- Less than 20 actions attempted
- Different strategies haven't been fully explored
- Page contains relevant content that could lead to success

**Termination Indicators** (ONLY WHEN ABSOLUTELY CERTAIN):
- ANY blocking condition detected (highest priority)
- Semantic goals are DEFINITIVELY satisfied (user intent completely fulfilled)
- SYSTEMATIC failures across ALL possible approaches
- NO viable pathways visible after extensive exploration
- Severe repetitive loops without ANY semantic progress
- Current state IMPOSSIBLE to lead to user goal satisfaction

**DECISION CRITERIA**:

**CONTINUE**: ANY progress evident, alternatives available, goal potentially achievable, less than 18 actions, NO blocking conditions
**TERMINATE**: ONLY when blocking conditions are confirmed, intent is definitively satisfied, or systematic failures across approaches with NO viable alternatives

**INSTRUCTION**: Focus on blocking condition detection first, then user intent satisfaction and execution viability.

**RESPONSE FORMAT**:
- To continue: `CONTINUE`
- To terminate: `TERMINATE: [Detailed justification for task completion, summarizing actions and current page state]`
"""


def build_blocking_analysis_prompt(failure_type: str, error_message: str, recent_failures: int, total_actions: int, page_indicators: dict) -> str:
    return f"""
        Analyze this web automation failure to determine if it represents a blocking condition that should terminate the task:

        **Failure Context:**
        - Failure Type: {failure_type}
        - Error Message: {error_message}
        - Recent Similar Failures: {recent_failures}
        - Total Actions Taken: {total_actions}

        **Page Context:**
        - URL: {page_indicators.get('url', 'unknown')}
        - Title: {page_indicators.get('title', 'unknown')}
        - Has Error Message: {page_indicators.get('has_error_message', False)}
        - Has Login Prompt: {page_indicators.get('has_login_prompt', False)}

        **Analysis Criteria:**
        1. **Immediate Blocking**: Access denied, authentication walls, regional restrictions
        2. **Repeated Failures**: Same failure type occurring multiple times without progress
        3. **System Errors**: Server errors, network issues, page not found
        4. **Recovery Potential**: Whether the agent can reasonably recover from this failure

        Respond with either "TERMINATE" if this represents a blocking condition that should end the task, 
        or "CONTINUE" if the agent should attempt to recover and continue.

        Focus on the semantic meaning and context rather than specific keywords.
        """


def build_completion_verification_prompt(task: str, action_journey: str, page_title: str, current_url: str, checklist_context: str = "") -> str:
    return f"""
You are an expert task completion verifier analyzing a complete web automation session.

**TASK**
{task}

**COMPLETE ACTION JOURNEY**
{action_journey}

**FINAL PAGE STATE**
Title: {page_title}
URL: {current_url}

        {checklist_context}

**VERIFICATION PROCESS**

1. **Identify ALL Task Requirements**
   Extract every explicit and implicit requirement from the task statement.
   Examples:
   - "Get women's black swimsuit, size large, highest discount, add to cart"
     → Requirements: (1) Women's category, (2) Black color, (3) Swimsuit type, (4) Size large, (5) Highest discount, (6) Added to cart

2. **Trace Action Journey**
   Follow the complete action sequence to verify each requirement was addressed:
   - Were the right pages visited?
   - Were appropriate filters/selections made?
   - Was there a logical progression toward the goal?

3. **Verify Final Page Evidence**
   Check if the FINAL PAGE shows:
   - ✓ Confirmation messages (e.g., "Added to basket", "Order confirmed", "Results shown")
   - ✓ Expected results visible (correct items, data displayed, forms submitted)
   - ✓ Success indicators (checkmarks, completion screens, cart updates)
   - ✗ Error messages or failed actions

4. **Apply Strict Completion Criteria**
   Task is COMPLETED only if:
   - ALL requirements were addressed in actions
   - Final page shows clear evidence of completion
   - No additional user actions needed
   - A human would consider this task done

   Task is NOT_COMPLETED if:
   - ANY requirement missing or not addressed
   - Final page doesn't confirm completion
   - User would need to do more work
   - There's any doubt about full completion

Respond with exactly one word: COMPLETED or NOT_COMPLETED
"""


def build_description_prompt(task: str, page_title: str, current_url: str, total_actions: int, action_summary: str) -> str:
    return f"""
You are tasked with providing a natural, conversational description of the current state of a web automation task. The agent has not signaled completion, so the task is still in progress.

**TASK CONTEXT**
Original Task: {task}
Current Page: {page_title} ({current_url})
Total Actions Taken: {total_actions}

**COMPLETE ACTION HISTORY**
{action_summary}

**INSTRUCTIONS**
Write a natural, conversational description that explains:
1. What the agent has accomplished so far
2. What the current page shows or represents
3. What progress has been made toward the task goal
4. What the agent might be working on or encountering

Write in a natural, flowing style as if you're explaining to someone what happened during this web automation session. Don't use bullet points, structured formats, or technical jargon. Just describe the journey and current state in plain, natural language.

Keep the description informative but concise (2-4 sentences). Focus on the actual progress and current situation rather than technical details.
"""


def build_evaluation_prompt(task: str, action_journey: str, page_title: str, current_url: str, total_actions: int) -> str:
    return f"""
You are an expert task completion evaluator analyzing a web automation session.

**TASK**
{task}

**COMPLETE ACTION JOURNEY** ({total_actions} steps total)
{action_journey}

**FINAL PAGE STATE**
Title: {page_title}
URL: {current_url}

**EVALUATION CRITERIA**

Analyze the COMPLETE journey and FINAL page state to determine task completion:

1. **Task Requirements Analysis**
   - Break down the task into specific requirements
   - Check if EACH requirement was addressed in the action sequence
   - Verify the final page confirms completion of ALL requirements

2. **Action Sequence Validation**
   - Did actions logically progress toward the goal?
   - Were all necessary steps performed (search, filter, select, submit, etc.)?
   - Were there confirmation actions (e.g., "Add to cart" → cart confirmation)?

3. **Final Page Evidence**
   - Does the visible page show SUCCESS indicators (confirmation messages, success modals, completion screens)?
   - Are the requested items/information visible and correct?
   - Is there evidence of the task outcome (items in cart, search results displayed, forms submitted)?

4. **Completion Signals**
   - Did the agent receive explicit confirmation (modal, message, redirect)?
   - Is the current page state consistent with a completed task?
   - Would a human user consider this task done based on what's visible?

**DECISION FRAMEWORK**
- **success**: ALL requirements met, final page shows clear completion evidence, logical action sequence
- **partial**: SOME requirements met OR task progressed significantly but not fully complete
- **failure**: FEW/NO requirements met OR no meaningful progress toward goal

Provide your evaluation in 2-4 sentences analyzing the complete journey and final state, then one line:
STATUS: success | partial | failure
"""


def build_summary_prompt(summary_source: str) -> str:
    return (
        "Summarize the earlier actions precisely and briefly. "
        "Cover what was done and outcomes. "
        "Do not include planned next steps. "
        "Limit to 3-5 sentences.\n\n" + summary_source
    )


def build_summary_update_prompt(old_summary: str, new_actions: str) -> str:
    return (
        "Update the earlier-actions summary concisely (2-3 sentences) "
        "by incorporating the new action details. "
        "Focus on what has been accomplished. "
        "Drop outdated details.\n\n"
        f"Old Summary: {old_summary}\n"
        f"New Actions: {new_actions}"
    )
