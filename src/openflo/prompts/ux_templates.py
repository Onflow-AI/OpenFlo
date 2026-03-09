# -*- coding: utf-8 -*-
"""
UX Evaluation Prompt Templates for SEQ to SUS conversion.

This module provides prompt builders for:
1. SEQ (Single Ease Question) evaluation per action
2. SUS (System Usability Scale) synthesis from SEQ data
3. Friction point analysis
"""

import json


def build_seq_evaluation_prompt(
    task_description: str,
    action_data: dict,
    page_context: dict,
    action_index: int,
    total_actions: int,
    custom_prompt: str = None
) -> str:
    """
    Build prompt for SEQ evaluation of a single action.

    Args:
        task_description: The overall task being performed
        action_data: Action record with type, description, success, etc.
        page_context: Current page URL and title
        action_index: 0-based index of this action
        total_actions: Total number of actions so far
        custom_prompt: Optional custom SEQ prompt template from config

    Returns:
        Prompt string for LLM to evaluate SEQ score
    """
    action_type = action_data.get('action', action_data.get('predicted_action', 'UNKNOWN'))
    action_desc = action_data.get('action_description', '')
    action_value = action_data.get('predicted_value', '')
    element_desc = action_data.get('element_description', '')
    success = action_data.get('success', True)
    error = action_data.get('error', '')
    action_generation = action_data.get('action_generation_response', '')

    # Use custom prompt if provided, otherwise use default
    if custom_prompt:
        base_prompt = custom_prompt
    else:
        base_prompt = """# SYSTEM PROMPT: BUA EXECUTION & MULTI-METRIC UX EVALUATION

## ROLE
You are an Autonomous Browser Use Agent (BUA) equipped with a UX Evaluation layer. Your goal is to execute a defined user flow while meticulously evaluating the user experience across multiple dimensions for every action.

## OPERATIONAL MANDATE
After completing every discrete action, you MUST evaluate FOUR metrics that capture different aspects of the user experience.

## EVALUATION METRICS (1-7 Scale)

You must evaluate FOUR dimensions for each action:

### 1. SEQ (Single Ease Question): Overall ease of completing this action
   - **7 = Very Easy**: Effortless, intuitive, immediate success
   - **6 = Easy**: Straightforward with clear path
   - **5 = Somewhat Easy**: Minor hesitation but generally clear
   - **4 = Neutral**: Required some thought or exploration
   - **3 = Somewhat Difficult**: Confusion or unclear path
   - **2 = Difficult**: Significant obstacles or errors
   - **1 = Very Difficult**: Extreme frustration, blocking errors, or complete confusion

### 2. Efficiency: Speed and directness of the action
   - **7 = Highly Efficient**: Instant, direct path, no delays
   - **6 = Efficient**: Quick with minimal steps
   - **5 = Moderately Efficient**: Some minor delays or extra steps
   - **4 = Neutral**: Average speed and directness
   - **3 = Somewhat Inefficient**: Noticeable delays or indirect path
   - **2 = Inefficient**: Excessive waiting, scrolling, or retrying
   - **1 = Very Inefficient**: Severe delays, multiple failed attempts, extensive searching

   **For efficiency, provide BOTH:**
   - A numeric score (1-7) based on the rubric above
   - A 1-2 sentence qualitative assessment describing what made the action efficient or inefficient (e.g., specific delays, number of steps, page responsiveness)

### 3. Clarity: How clear and understandable the UI element/feedback was
   - **7 = Crystal Clear**: Obvious, well-labeled, immediate and clear feedback
   - **6 = Very Clear**: Easy to understand with good labeling
   - **5 = Clear**: Understandable with minor ambiguity
   - **4 = Neutral**: Adequate but not exceptional clarity
   - **3 = Somewhat Unclear**: Some ambiguity or poor labeling
   - **2 = Unclear**: Confusing, hidden, or ambiguous elements
   - **1 = Very Unclear**: Extremely ambiguous, no feedback, or completely hidden

   **For clarity, provide BOTH:**
   - A numeric score (1-7) based on the rubric above
   - A 1-2 sentence qualitative assessment describing the UI clarity (e.g., labeling quality, visual feedback, element visibility)

### 4. Confidence: User's certainty about the action and its outcome
   - **7 = Very Confident**: Completely certain about what to do and what will happen
   - **6 = Confident**: Strong sense of correctness
   - **5 = Mostly Confident**: Minor uncertainty but generally sure
   - **4 = Neutral**: Uncertain but willing to proceed
   - **3 = Somewhat Uncertain**: Noticeable doubt about correctness
   - **2 = Uncertain**: Guessing or unsure of outcome
   - **1 = Very Uncertain**: Complete uncertainty, high risk of error

   **For confidence, provide BOTH:**
   - A numeric score (1-7) based on the rubric above
   - A 1-2 sentence qualitative assessment describing the level of certainty (e.g., whether the action path was obvious, if there was doubt about the outcome)

## EVALUATION GUIDELINES

### Thinking Log Requirements:
- Identify specific DOM elements interacted with
- Note any latency, layout shifts, or visual noise
- Mention if you had to backtrack or re-scan the page
- Explain how efficiency, clarity, and confidence were affected
- Provide a unified explanation that covers all four metrics

### Friction Indicators:
Include any applicable friction types: "waiting", "searching", "retrying", "scrolling", "confusion", "error", "ambiguity", "uncertainty"

## SCORING GUIDANCE: USE THE FULL SCALE

**CRITICAL**: Avoid binary thinking. Use the full 1-7 scale to capture nuances in user experience. Each score point represents a meaningful difference.

### Common Scenarios and Appropriate Scores:

**Score 7 (Perfect)**:
- Element loads instantly (<100ms)
- Label perfectly matches expected action
- Zero hesitation or searching required
- Example: Clicking a prominently placed "Submit" button that works immediately

**Score 6 (Very Good)**:
- Minor delay (100-500ms) but no impact on flow
- Clear label with good visual hierarchy
- Minimal searching (<2 seconds)
- Example: Finding a "Sign In" link in header after brief scan

**Score 5 (Good with Minor Issues)**:
- Noticeable delay (500ms-2s) or 1 extra click required
- Label is clear but not immediately obvious
- Some searching required (2-5 seconds)
- Example: Scrolling down to find a footer link that's clearly labeled once found

**Score 4 (Neutral/Mixed)**:
- Moderate delay (2-4s) or 2-3 extra steps
- Ambiguous labeling requiring interpretation
- Trial-and-error or exploring multiple options
- Example: Unclear navigation structure requiring checking 2-3 menu items

**Score 3 (Problematic but Workable)**:
- Significant delay (4-8s) or indirect path requiring backtracking
- Poor labeling, had to infer meaning from context
- Extensive searching or multiple attempts
- Example: Form validation error message is vague but eventually understood

**Score 2 (Severe Issues but Eventually Succeeded)**:
- Very long delay (8-15s) or multiple failed attempts before success
- Confusing UI requiring guesswork
- Had to retry 2-3 times or use workarounds
- Example: Button requires multiple clicks to work, or found element through trial-and-error

**Score 1 (Blocking Failure)**:
- Complete inability to proceed
- Element entirely missing or completely broken
- No path forward without intervention
- Example: Target element doesn't exist, critical error prevents action

### Error Handling - Use Graduated Scoring:

**For FAILED actions**, score based on the REASON for failure:
- **Complete blocking error** (missing element, system crash): Score 1
- **Temporary failure that was overcome** (retry succeeded, workaround found): Score 2-3
- **Partial success** (wrong element clicked but close): Score 3-4

**For SUCCESSFUL actions**, score based on QUALITY of experience:
- **Don't automatically assign 7** just because action succeeded
- Consider delays, extra steps, confusion along the way
- Success with friction = scores 3-6 depending on severity

### Independent Metric Scoring:

**Each metric can have different scores** for the same action:
- High efficiency (7) + Low clarity (3) = Fast but confusing
- Low efficiency (2) + High clarity (7) = Slow but obvious
- High confidence (6) + Medium efficiency (4) = Sure but requires extra steps

Example: Cookie banner with small "X" button
- SEQ: 4 (neutral - works but annoying)
- Efficiency: 3 (had to search for tiny button)
- Clarity: 5 (X icon is standard but very small)
- Confidence: 6 (confident it will close, just hard to click)
"""

    # Append context data
    context_section = f"""

## CURRENT TASK CONTEXT
**Overall Task**: {task_description}
**Current Step**: {action_index + 1} of {total_actions}

## ACTION DETAILS
- **Action Type**: {action_type}
- **Action Description**: {action_desc}
- **Value/Input**: {action_value if action_value else 'N/A'}
- **Target Element**: {element_desc if element_desc else 'N/A'}
- **Outcome**: {'Successful' if success else 'Failed - ' + str(error)}

## AGENT'S REASONING
{action_generation[:500] if action_generation else 'No reasoning recorded'}

## PAGE CONTEXT
- **URL**: {page_context.get('url', 'Unknown')}
- **Page Title**: {page_context.get('title', 'Unknown')}

## RESPONSE FORMAT (JSON)
```json
{{
    "seq_score": <1-7>,
    "efficiency": <1-7>,
    "efficiency_assessment": "<1-2 sentences describing speed/directness>",
    "clarity": <1-7>,
    "clarity_assessment": "<1-2 sentences describing UI clarity>",
    "confidence": <1-7>,
    "confidence_assessment": "<1-2 sentences describing certainty>",
    "thinking_log": "<unified explanation covering all four metrics - be specific about DOM elements, timing, clarity, and certainty>",
    "friction_indicators": ["<list any: waiting, searching, retrying, scrolling, confusion, error, ambiguity, uncertainty>"]
}}
```

Evaluate this action across all four dimensions:"""

    return base_prompt + context_section


def build_sus_synthesis_prompt(
    task_description: str,
    seq_data: list,
    session_summary: str,
    average_seq: float
) -> str:
    """
    Build prompt for SUS synthesis using the provided system prompt specification.

    Args:
        task_description: The task that was executed
        seq_data: List of SEQ evaluation results
        session_summary: Natural language summary of the session
        average_seq: Average SEQ score across all steps

    Returns:
        Prompt string for LLM to generate SUS evaluation
    """
    # Format multi-metric SEQ data for prompt
    seq_entries = []
    for seq in seq_data:
        thinking = seq.get('thinking_log', 'N/A')
        if len(thinking) > 200:
            thinking = thinking[:200] + "..."

        # Get qualitative assessments (truncate if too long)
        eff_assessment = seq.get('efficiency_assessment', '')
        if len(eff_assessment) > 60:
            eff_assessment = eff_assessment[:60] + "..."

        clarity_assessment = seq.get('clarity_assessment', '')
        if len(clarity_assessment) > 60:
            clarity_assessment = clarity_assessment[:60] + "..."

        conf_assessment = seq.get('confidence_assessment', '')
        if len(conf_assessment) > 60:
            conf_assessment = conf_assessment[:60] + "..."

        seq_entries.append(
            f"Step {seq.get('step', '?')}: "
            f"SEQ={seq.get('seq_score', '?')}/7 | "
            f"Efficiency={seq.get('efficiency', '?')}/7 ('{eff_assessment}') | "
            f"Clarity={seq.get('clarity', '?')}/7 ('{clarity_assessment}') | "
            f"Confidence={seq.get('confidence', '?')}/7 ('{conf_assessment}') | "
            f"Action: {seq.get('action_type', 'Unknown')} | "
            f"Success: {seq.get('success', True)} | "
            f"Thinking: {thinking}"
        )
    seq_text = "\n".join(seq_entries)

    # Analyze SEQ patterns for heuristic hints
    scores = [s.get('seq_score', 4) for s in seq_data]
    pattern_hints = _analyze_seq_patterns(scores, seq_data)

    return f"""# UX SYNTHESIS AGENT (SEQ TO SUS)

## ROLE
You are a UX Research & Data Synthesis Agent. Your role is to analyze a completed user flow session where FOUR metrics were recorded after every step (SEQ, Efficiency, Clarity, Confidence), and transform those micro-metrics into a macro-level System Usability Scale (SUS) report.

## OBJECTIVE
Evaluate the overall usability of the tested user flow by mapping the multi-metric scores (1-7 scale) and the corresponding thinking logs to the 10-item SUS framework (1-5 scale).

## INPUT DATA

### Task Description
{task_description}

### Session Summary
{session_summary}

### Average SEQ Score
{average_seq:.2f}/7

### SEQ Pattern Analysis
{pattern_hints}

### Step-by-Step Multi-Metric Data
{seq_text}

## ENHANCED MAPPING LOGIC (Multi-Metric → SUS)

### PRIMARY RULE: USE AVERAGE SCORES, NOT OUTLIERS
**CRITICAL**: Base your SUS evaluation on the AVERAGE scores across all metrics, not individual low-score outliers. A few difficult steps in an otherwise smooth experience should NOT result in universally negative SUS scores.

### Primary Metric: SEQ Score
SEQ drives the base assessment for all 10 SUS items:
- **High average SEQ (≥5.0)**: Generally positive SUS (items 1,3,5,7,9 score 4-5; items 2,4,6,8,10 score 1-2)
- **Moderate average SEQ (4.0-4.9)**: Mixed SUS (items score around 3, adjusted by modifier metrics)
- **Low average SEQ (<4.0)**: Generally negative SUS (items 1,3,5,7,9 score 1-2; items 2,4,6,8,10 score 4-5)
- Volatile SEQ scores: High Inconsistency (Item 6 should be HIGH)
- Improving SEQ over time: High Learnability (Item 7 should be HIGH)

### Modifier Metrics: Efficiency, Clarity, Confidence

Each modifier metric now includes both a numeric score (1-7) AND a qualitative assessment (1-2 sentences). **Use both the numeric scores and the qualitative descriptions** to understand the nuanced context behind each metric when mapping to SUS items.

**Efficiency** specifically modifies:
- **Item 8 (Cumbersomeness)**: Low efficiency scores → HIGH cumbersomeness score
  - Example: Average efficiency < 3 → Item 8 should be 4-5 (system IS cumbersome)
- **Item 7 (Learnability)**: Improving efficiency over time → HIGH learnability score
  - Example: Efficiency rises from 2 to 6 → Item 7 should be 4-5 (people learn quickly)

**Clarity** specifically modifies:
- **Item 2 (Complexity)**: Low clarity → HIGH complexity score
  - Example: Average clarity < 3 → Item 2 should be 4-5 (system IS complex)
- **Item 3 (Ease of use)**: High clarity → HIGH ease score
  - Example: Average clarity > 5 → Item 3 should be 4-5 (system IS easy)
- **Item 4 (Need support)**: Low clarity → HIGH need for support
  - Example: Average clarity < 3 → Item 4 should be 4-5 (WOULD need support)

**Confidence** specifically modifies:
- **Item 9 (User confidence)**: High confidence → HIGH confidence score
  - Example: Average confidence > 5 → Item 9 should be 4-5 (felt confident)
- **Item 1 (Frequency of use)**: Low confidence → LOW frequency score
  - Example: Average confidence < 3 → Item 1 should be 1-2 (would NOT use frequently)

### Practical Examples:
- SEQ=2, Efficiency=1, Clarity=2 → Item 8 (Cumbersome) should be 5 (Strongly Agree)
- SEQ=6, Clarity=7, Confidence=7 → Item 3 (Easy to use) should be 5 (Strongly Agree)
- SEQ=5, Clarity=2, Confidence=2 → Item 3 should be 2-3 (despite moderate SEQ)
- SEQ=4, Efficiency=7, Clarity=6 → System is reasonably usable despite neutral SEQ

**CRITICAL**: When a specific metric contradicts the SEQ, ALWAYS give weight to the specialized metric for its corresponding SUS items. The modifier metrics provide nuanced insight that SEQ alone cannot capture.

## THE SUS EVALUATION (1-5 Scale)
Assign a score from 1 (Strongly Disagree) to 5 (Strongly Agree) for each item.

**IMPORTANT**: Items 1, 3, 5, 7, 9 are POSITIVE statements (higher = better usability).
Items 2, 4, 6, 8, 10 are NEGATIVE statements (higher = worse usability).

1. I think that I would like to use this system frequently.
2. I found the system unnecessarily complex.
3. I thought the system was easy to use.
4. I think that I would need the support of a technical person to be able to use this system.
5. I found the various functions in this system were well integrated.
6. I thought there was too much inconsistency in this system.
7. I would imagine that most people would learn to use this system very quickly.
8. I found the system very cumbersome to use.
9. I felt very confident using the system.
10. I needed to learn a lot of things before I could get going with this system.

## OUTPUT FORMAT (JSON)

**REMINDER**: Your scores should reflect the OVERALL experience based on AVERAGE metrics. If 75% of steps scored 6-7, the overall experience is POSITIVE regardless of a few low-scoring outliers.

```json
{{
    "sus_items": [
        {{"id": 1, "text": "I think that I would like to use this system frequently.", "score": <1-5>, "rationale": "<explain based on AVERAGE SEQ and confidence>"}},
        {{"id": 2, "text": "I found the system unnecessarily complex.", "score": <1-5>, "rationale": "<explain based on AVERAGE clarity>"}},
        {{"id": 3, "text": "I thought the system was easy to use.", "score": <1-5>, "rationale": "<explain based on AVERAGE SEQ and clarity>"}},
        {{"id": 4, "text": "I think that I would need the support of a technical person to be able to use this system.", "score": <1-5>, "rationale": "<explain based on AVERAGE clarity>"}},
        {{"id": 5, "text": "I found the various functions in this system were well integrated.", "score": <1-5>, "rationale": "<explain based on AVERAGE SEQ>"}},
        {{"id": 6, "text": "I thought there was too much inconsistency in this system.", "score": <1-5>, "rationale": "<explain based on score VOLATILITY>"}},
        {{"id": 7, "text": "I would imagine that most people would learn to use this system very quickly.", "score": <1-5>, "rationale": "<explain based on SEQ TRENDS>"}},
        {{"id": 8, "text": "I found the system very cumbersome to use.", "score": <1-5>, "rationale": "<explain based on AVERAGE efficiency>"}},
        {{"id": 9, "text": "I felt very confident using the system.", "score": <1-5>, "rationale": "<explain based on AVERAGE confidence>"}},
        {{"id": 10, "text": "I needed to learn a lot of things before I could get going with this system.", "score": <1-5>, "rationale": "<explain based on initial learning curve>"}}
    ],
    "mapping_analysis": "<2-3 sentence summary emphasizing AVERAGE scores and overall experience quality>"
}}
```

Generate the SUS evaluation:"""


def build_friction_analysis_prompt(seq_data: list, low_score_threshold: int = 3) -> str:
    """
    Build prompt for identifying key friction points from low SEQ scores.

    Args:
        seq_data: List of SEQ evaluation results
        low_score_threshold: Threshold for considering a score "low" (default 3)

    Returns:
        Prompt string for friction analysis
    """
    low_scores = [s for s in seq_data if s.get('seq_score', 7) <= low_score_threshold]

    entries = []
    for seq in low_scores:
        friction = ", ".join(seq.get('friction_indicators', [])) or "None identified"
        entries.append(
            f"- Step {seq.get('step', '?')}: SEQ={seq.get('seq_score', '?')}/7\n"
            f"  Action: {seq.get('action_type', 'Unknown')}\n"
            f"  Element: {seq.get('action_description', 'N/A')}\n"
            f"  Thinking: {seq.get('thinking_log', 'N/A')}\n"
            f"  Friction indicators: {friction}"
        )

    entries_text = "\n\n".join(entries) if entries else "No low-scoring steps identified."

    return f"""# Friction Point Analysis

Analyze the following low-scoring steps (SEQ <= {low_score_threshold}) to identify the most significant UX friction points.

## Low-Scoring Steps
{entries_text}

## Instructions
1. Identify the root cause of each low score
2. Categorize friction types:
   - **Findability**: Element hard to locate
   - **Clarity**: Unclear labels or purpose
   - **Feedback**: Poor system response
   - **Complexity**: Too many steps or options
   - **Performance**: Slow loading or response
   - **Obstruction**: Popups, overlays blocking action
3. Rank by severity/impact on user experience
4. Suggest actionable improvements

## Output Format (JSON)
```json
{{
    "friction_points": [
        {{
            "step": <step_number>,
            "severity": "<high|medium|low>",
            "category": "<findability|clarity|feedback|complexity|performance|obstruction>",
            "description": "<specific description of what caused the friction>",
            "recommendation": "<actionable improvement suggestion>"
        }}
    ],
    "overall_assessment": "<1-2 sentence summary of main usability issues>"
}}
```

Analyze the friction points:"""


def _analyze_seq_patterns(scores: list, seq_data: list) -> str:
    """
    Analyze all metric patterns (SEQ, Efficiency, Clarity, Confidence) to provide hints for SUS mapping.

    Args:
        scores: List of SEQ scores
        seq_data: Full SEQ data with all metrics and thinking logs

    Returns:
        Pattern analysis string
    """
    if not scores:
        return "No scores to analyze."

    patterns = []

    # Extract all metrics
    efficiency_scores = [s.get('efficiency', 4) for s in seq_data]
    clarity_scores = [s.get('clarity', 4) for s in seq_data]
    confidence_scores = [s.get('confidence', 4) for s in seq_data]

    # Calculate overall averages FIRST to establish baseline
    avg_seq = sum(scores) / len(scores)
    avg_eff = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 4
    avg_clarity = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 4
    avg_conf = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 4

    # Add POSITIVE baseline patterns first (most important signal)
    if avg_seq >= 5.5:
        patterns.append(f"- **Overall SEQ**: High average ({avg_seq:.1f}/7) - system is generally easy to use")
    elif avg_seq >= 4.5:
        patterns.append(f"- **Overall SEQ**: Moderate-high average ({avg_seq:.1f}/7) - system is reasonably usable")
    elif avg_seq >= 3.5:
        patterns.append(f"- **Overall SEQ**: Neutral average ({avg_seq:.1f}/7) - mixed usability")
    else:
        patterns.append(f"- **Overall SEQ**: Low average ({avg_seq:.1f}/7) - significant usability issues")

    # SEQ patterns: Check for learning curve (only if meaningful change)
    if len(scores) >= 3:
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]
        first_avg = sum(first_half) / len(first_half) if first_half else 0
        second_avg = sum(second_half) / len(second_half) if second_half else 0
        if second_avg > first_avg + 1.0:  # Stricter threshold: need 1+ point improvement
            patterns.append("- **SEQ Trend**: Significant improvement over time (good learnability)")
        elif first_avg > second_avg + 1.0:  # Stricter threshold: need 1+ point decline
            patterns.append("- **SEQ Trend**: Declined over time (possible fatigue or complexity)")

    # SEQ patterns: Check for volatility (stricter threshold)
    if len(scores) >= 3:
        variance = sum((s - avg_seq)**2 for s in scores) / len(scores)
        if variance > 4.0:  # Increased threshold from 2.5 to 4.0
            patterns.append("- **SEQ Volatility**: Large score swings detected (inconsistent experience)")

    # SEQ patterns: Check for consistently low scores (stricter threshold)
    low_count = sum(1 for s in scores if s <= 3)
    high_count = sum(1 for s in scores if s >= 6)
    if low_count > len(scores) * 0.5:  # Majority must be low
        patterns.append(f"- **SEQ Distribution**: Majority low - {low_count}/{len(scores)} steps scored ≤3")
    elif high_count > len(scores) * 0.5:  # Add positive pattern for majority high
        patterns.append(f"- **SEQ Distribution**: Majority high - {high_count}/{len(scores)} steps scored ≥6")

    # Efficiency patterns: Add positive patterns first
    if avg_eff >= 5.5:
        patterns.append(f"- **Efficiency**: High average ({avg_eff:.1f}/7) - actions were fast and direct")
    elif avg_eff < 3.0:
        patterns.append(f"- **Efficiency**: Low average ({avg_eff:.1f}/7) - system felt cumbersome")

    if len(efficiency_scores) >= 3:
        eff_first = efficiency_scores[:len(efficiency_scores)//2]
        eff_second = efficiency_scores[len(efficiency_scores)//2:]
        eff_first_avg = sum(eff_first) / len(eff_first) if eff_first else 0
        eff_second_avg = sum(eff_second) / len(eff_second) if eff_second else 0
        if eff_second_avg > eff_first_avg + 1.0:  # Stricter threshold
            patterns.append("- **Efficiency Trend**: Improved significantly over time")

    # Clarity patterns: Add positive patterns first
    if avg_clarity >= 5.5:
        patterns.append(f"- **Clarity**: High average ({avg_clarity:.1f}/7) - UI was clear and well-labeled")
    elif avg_clarity < 3.0:
        patterns.append(f"- **Clarity**: Low average ({avg_clarity:.1f}/7) - UI elements were unclear")
    else:
        clarity_low_count = sum(1 for c in clarity_scores if c <= 3)
        # Only flag if majority (>50%) had clarity issues, not just 40%
        if clarity_low_count > len(clarity_scores) * 0.5:
            patterns.append(f"- **Clarity Issues**: {clarity_low_count}/{len(clarity_scores)} steps had low clarity")

    # Confidence patterns: Add positive patterns first
    if avg_conf >= 5.5:
        patterns.append(f"- **Confidence**: High average ({avg_conf:.1f}/7) - users felt certain about actions")
    elif avg_conf < 3.0:
        patterns.append(f"- **Confidence**: Low average ({avg_conf:.1f}/7) - high uncertainty")
    else:
        conf_variance = sum((c - avg_conf)**2 for c in confidence_scores) / len(confidence_scores)
        if conf_variance > 4.0:  # Stricter threshold (was 2.5)
            patterns.append("- **Confidence Volatility**: Certainty fluctuated significantly")

    # Check for friction indicators in thinking logs (stricter threshold)
    friction_keywords = ['wait', 'search', 'retry', 'confus', 'scroll', 'hidden', 'unclear', 'ambiguous', 'uncertain']
    friction_count = 0
    for seq in seq_data:
        thinking = (seq.get('thinking_log', '') + ' '.join(seq.get('friction_indicators', []))).lower()
        if any(kw in thinking for kw in friction_keywords):
            friction_count += 1
    # Only flag if majority (>50%) had friction, not just 30%
    if friction_count > len(seq_data) * 0.5:
        patterns.append(f"- **Friction Indicators**: {friction_count}/{len(seq_data)} steps had friction keywords")

    return "\n".join(patterns) if patterns else "- No significant patterns detected"


def parse_seq_response(response: str) -> dict:
    """
    Parse LLM response to extract all four metrics: SEQ, efficiency, clarity, confidence.

    Args:
        response: Raw LLM response string

    Returns:
        Parsed dict with seq_score, efficiency, clarity, confidence, thinking_log, friction_indicators,
        plus qualitative assessments (efficiency_assessment, clarity_assessment, confidence_assessment)
    """
    import re

    # Fallback qualitative descriptions based on numeric scores
    QUALITATIVE_FALLBACK = {
        "efficiency": {
            7: "Instant and direct action with no delays.",
            6: "Quick action with minimal steps required.",
            5: "Reasonably fast with some minor delays observed.",
            4: "Average speed and directness for the action.",
            3: "Noticeable delays or indirect path taken.",
            2: "Excessive waiting, scrolling, or retrying required.",
            1: "Severe delays with multiple failed attempts."
        },
        "clarity": {
            7: "Crystal clear UI with obvious labels and immediate feedback.",
            6: "Very clear interface with good labeling.",
            5: "Clear and understandable with minor ambiguity.",
            4: "Adequate clarity but not exceptional.",
            3: "Some ambiguity or poor labeling present.",
            2: "Confusing or hidden UI elements.",
            1: "Extremely ambiguous with no clear feedback."
        },
        "confidence": {
            7: "Completely certain about the action and expected outcome.",
            6: "Strong sense of correctness and confidence.",
            5: "Mostly confident with minor uncertainty.",
            4: "Uncertain but willing to proceed.",
            3: "Noticeable doubt about the correctness of the action.",
            2: "Guessing or unsure about the outcome.",
            1: "Complete uncertainty with high risk of error."
        }
    }

    # Default values (neutral for all metrics)
    result = {
        "seq_score": 4,
        "efficiency": 4,
        "efficiency_assessment": "",
        "clarity": 4,
        "clarity_assessment": "",
        "confidence": 4,
        "confidence_assessment": "",
        "thinking_log": "",
        "friction_indicators": []
    }

    if not response:
        return result

    # Try to extract JSON from response
    try:
        # Find JSON block in response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            parsed = json.loads(json_match.group())
            result["seq_score"] = int(parsed.get("seq_score", 4))
            result["efficiency"] = int(parsed.get("efficiency", 4))
            result["clarity"] = int(parsed.get("clarity", 4))
            result["confidence"] = int(parsed.get("confidence", 4))
            result["thinking_log"] = str(parsed.get("thinking_log", ""))
            result["friction_indicators"] = list(parsed.get("friction_indicators", []))

            # Extract qualitative assessments if present
            result["efficiency_assessment"] = str(parsed.get("efficiency_assessment", ""))
            result["clarity_assessment"] = str(parsed.get("clarity_assessment", ""))
            result["confidence_assessment"] = str(parsed.get("confidence_assessment", ""))

            # Clamp all scores to valid range (1-7)
            result["seq_score"] = max(1, min(7, result["seq_score"]))
            result["efficiency"] = max(1, min(7, result["efficiency"]))
            result["clarity"] = max(1, min(7, result["clarity"]))
            result["confidence"] = max(1, min(7, result["confidence"]))

            # Apply fallback for missing qualitative assessments
            if not result["efficiency_assessment"]:
                result["efficiency_assessment"] = QUALITATIVE_FALLBACK["efficiency"].get(
                    result["efficiency"], "Assessment unavailable"
                )
            if not result["clarity_assessment"]:
                result["clarity_assessment"] = QUALITATIVE_FALLBACK["clarity"].get(
                    result["clarity"], "Assessment unavailable"
                )
            if not result["confidence_assessment"]:
                result["confidence_assessment"] = QUALITATIVE_FALLBACK["confidence"].get(
                    result["confidence"], "Assessment unavailable"
                )
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fallback: try to extract scores from text
        score_match = re.search(r'seq_score["\s:]+(\d)', response, re.IGNORECASE)
        if score_match:
            result["seq_score"] = max(1, min(7, int(score_match.group(1))))

        efficiency_match = re.search(r'efficiency["\s:]+(\d)', response, re.IGNORECASE)
        if efficiency_match:
            result["efficiency"] = max(1, min(7, int(efficiency_match.group(1))))

        clarity_match = re.search(r'clarity["\s:]+(\d)', response, re.IGNORECASE)
        if clarity_match:
            result["clarity"] = max(1, min(7, int(clarity_match.group(1))))

        confidence_match = re.search(r'confidence["\s:]+(\d)', response, re.IGNORECASE)
        if confidence_match:
            result["confidence"] = max(1, min(7, int(confidence_match.group(1))))

        # Extract thinking log from text if present
        thinking_match = re.search(r'thinking_log["\s:]+["\'](.*?)["\']', response, re.IGNORECASE)
        if thinking_match:
            result["thinking_log"] = thinking_match.group(1)

    return result


def parse_sus_response(response: str) -> dict:
    """
    Parse LLM response to extract SUS items and mapping analysis.

    Args:
        response: Raw LLM response string

    Returns:
        Parsed dict with sus_items list and mapping_analysis
    """
    import re

    # Default values
    result = {
        "sus_items": [],
        "mapping_analysis": ""
    }

    if not response:
        return result

    try:
        # Find JSON block in response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            parsed = json.loads(json_match.group())
            sus_items = parsed.get("sus_items", [])

            # Validate and normalize sus_items
            for item in sus_items:
                if isinstance(item, dict) and "id" in item and "score" in item:
                    item["score"] = max(1, min(5, int(item.get("score", 3))))

            result["sus_items"] = sus_items
            result["mapping_analysis"] = str(parsed.get("mapping_analysis", ""))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return result


def parse_friction_response(response: str) -> dict:
    """
    Parse LLM response for friction analysis.

    Args:
        response: Raw LLM response string

    Returns:
        Parsed dict with friction_points list and overall_assessment
    """
    import re

    result = {
        "friction_points": [],
        "overall_assessment": ""
    }

    if not response:
        return result

    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            parsed = json.loads(json_match.group())
            result["friction_points"] = list(parsed.get("friction_points", []))
            result["overall_assessment"] = str(parsed.get("overall_assessment", ""))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return result
