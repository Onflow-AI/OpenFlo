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
import re
import shlex

def format_choices(elements):
    """
    Format elements into choices for the prompt, with robust error handling.
    Returns empty list if elements is None or empty.
    """
    if not elements:
        print("WARNING: format_choices received empty or None elements list")
        return []
        
    converted_elements = []
    elements_w_descriptions = []
    
    for element in elements:
        # Validate element structure
        if not element or not isinstance(element, dict):
            print(f"WARNING: Skipping invalid element: {element}")
            continue
            
        # Check for required fields
        required_fields = ["center_point", "tag_with_role", "description", "tag"]
        if not all(field in element for field in required_fields):
            print(f"WARNING: Element missing required fields: {element}")
            continue
            
        if "description" in element and "=" in element["description"] and "'" not in element["description"] and "\"" not in element["description"]:
            description_dict = [] 
            try:
                for sub in shlex.split(element["description"]): 
                    if '=' in sub:
                        key_value = sub.split('=', 1)
                        if len(key_value) == 2:
                            description_dict.append((key_value[0].strip(), key_value[1].strip()))
                element.update(dict(description_dict))
            except (ValueError, AttributeError) as e:
                print(f"WARNING: Error parsing element description: {e}")
        elements_w_descriptions.append(element)

    converted_elements = []
    for i, element in enumerate(elements_w_descriptions):
        try:
            converted = ""
            if element.get('tag') != "select":
                converted += f'{element.get("center_point", "")} <{element.get("tag_with_role", "")}>'
                converted += (
                    element.get("description", "")
                    if len((element.get("description") or "").split()) < 30
                    else " ".join((element.get("description") or "").split()[:30]) + "..."
                )
                converted += f"</{element.get('tag', '')}>"
            else:
                converted += f'{element.get("center_point", "")} <{element.get("tag_with_role", "")}>'
                converted += (
                    element.get("description", "")
                )
                converted += f"</{element.get('tag', '')}>"
            converted_elements.append(converted)
        except Exception as e:
            print(f"WARNING: Error formatting element {i}: {e}")
            continue

    print(f"INFO: Successfully formatted {len(converted_elements)} choices from {len(elements)} elements")
    return converted_elements

def _clean_llm_response_patterns(text):
    """
    Centralized function to clean common LLM response patterns.
    This eliminates code duplication between postprocess functions.
    """
    if not text:
        return ""
    
    text = text.strip()
    
    # Define patterns to clean - organized by category for better maintainability
    patterns_to_remove = [
        # Choice selection patterns
        "The uppercase letter of your choice. Choose one of the following elements if it matches the target element based on your analysis:\n\n",
        "The uppercase letter of your choice. Choose one of the following elements if it matches the target element based on your analysis:\n",
        "The uppercase letter of your choice. Choose one of the following elements if it matches the target element based on your analysis:",
        
        # Analysis-based choice patterns
        "The uppercase letter of your choice based on your analysis is:\n\n",
        "The uppercase letter of your choice based on your analysis is:\n", 
        "The uppercase letter of your choice based on your analysis is:",
        "The uppercase letter of your choice based on the analysis is:\n\n",
        "The uppercase letter of your choice based on the analysis is:\n",
        "The uppercase letter of your choice based on the analysis is:",
        "The uppercase letter of your choice based on the analysis is ",
        "The uppercase letter of your choice based on my analysis is:\n\n",
        "The uppercase letter of your choice based on my analysis is:\n",
        "The uppercase letter of your choice based on my analysis is:",
        
        # My choice patterns
        "The uppercase letter of my choice is \n\n",
        "The uppercase letter of my choice is \n",
        "The uppercase letter of my choice is ",
        "The uppercase letter of my choice is:\n\n",
        "The uppercase letter of my choice is:\n",
        "The uppercase letter of my choice is:",
        "The uppercase letter of my choice based on the analysis is:\n\n",
        "The uppercase letter of my choice based on the analysis is:\n",
        "The uppercase letter of my choice based on the analysis is:",
        "The uppercase letter of my choice based on the analysis is ",
        
        # Your choice patterns
        "The uppercase letter of your choice is \n\n",
        "The uppercase letter of your choice is \n",
        "The uppercase letter of your choice is ",
        "The uppercase letter of your choice.\n\n",
        "The uppercase letter of your choice.\n",
        "The uppercase letter of your choice.",
        
        # Correct choice patterns
        "The correct choice based on the analysis would be:\n\n",
        "The correct choice based on the analysis would be:\n",
        "The correct choice based on the analysis would be :",
        "The correct choice based on the analysis would be ",
        "The correct element to select would be:\n\n",
        "The correct element to select would be:\n",
        "The correct element to select would be:",
        "The correct element to select would be ",
        
        # Action instruction patterns
        "Choose an action from {CLICK, TYPE, SELECT}.\n\n",
        "Choose an action from {CLICK, TYPE, SELECT}.\n",
        "Choose an action from {CLICK, TYPE, SELECT}.",
        "Provide additional input based on ACTION.\n\n",
        "Provide additional input based on ACTION.\n",
        "Provide additional input based on ACTION.",
    ]
    
    # Apply all pattern removals
    for pattern in patterns_to_remove:
        text = text.replace(pattern, "")
    
    return text

def postprocess_action_lmm(text):
    """
    Enhanced postprocess function with better error handling and validation.
    Returns (element_label, action, value) with safe defaults.
    """
    try:
        if text is None:
            print("WARNING: postprocess_action_lmm received None text")
            return "None", "NONE", "None"
            
        if not isinstance(text, str):
            print(f"WARNING: postprocess_action_lmm received non-string input: {type(text)}")
            text = str(text) if text is not None else ""
    
        # Use centralized cleaning function
        text = _clean_llm_response_patterns(text)
        
        # Extract element selection with error handling
        selected_option = re.findall(r"ELEMENT: ([A-Z]{2}|[A-Z])", text)

        if selected_option:
            selected_option = (
                selected_option[0]
            )
        else:
            selected_option = "Invalid"

        # Extract action with error handling
        action = re.search(
            r"ACTION: (CLICK|SELECT|TYPE|HOVER|PRESS ENTER|SCROLL UP|SCROLL DOWN|SCROLL TOP|SCROLL BOTTOM|PRESS HOME|PRESS END|PRESS PAGEUP|PRESS PAGEDOWN|NEW TAB|CLOSE TAB|GO BACK|GO FORWARD|TERMINATE|NONE|GOTO|SAY)",
            text
        )

        if action:
            action = action.group(1)
            start = text.find(f"ACTION: {action}")
            for probing_length in range(15, 160, 10):
                selected_option_from_action = re.findall(
                    r"ELEMENT: ([A-Z]{2}|[A-Z])",
                    text[max(start - probing_length, 0):start])
                if selected_option_from_action and len(selected_option_from_action) > 0:
                    selected_option = selected_option_from_action[0]
                    break
        else:
            action = "None"

        # Extract value with error handling
        value = re.search(r"VALUE: (.*)$", text, re.MULTILINE)
        value = value.group(1) if value is not None else ""
        
        # Process and return results with safe defaults
        # Handle various forms of None/empty values
        if value:
            processed_value = process_string(process_string((value or "").strip()))
            # Check if processed value is effectively None
            if processed_value.lower() in ["none", "null", "", "n/a"]:
                processed_value = ""  # Use empty string instead of "None"
        else:
            processed_value = ""  # Use empty string instead of "None"
            
        return selected_option, (action or "").strip(), processed_value
        
    except Exception as e:
        print(f"ERROR in postprocess_action_lmm: {e}")
        print(f"Input text: {text}")
        # Return safe defaults on any error - use empty string instead of "None"
        return "None", "NONE", ""


def postprocess_action_lmm_pixel(text):
    # Use centralized cleaning function
    text = _clean_llm_response_patterns(text or "")

    action = re.search(
        r"ACTION: (CLICK|SELECT|TYPE|HOVER|PRESS ENTER|SCROLL UP|SCROLL DOWN|SCROLL TOP|SCROLL BOTTOM|PRESS HOME|PRESS END|PRESS PAGEUP|PRESS PAGEDOWN|NEW TAB|CLOSE TAB|GO BACK|GO FORWARD|TERMINATE|NONE|GOTO|SAY)",
        text
    )

    if action:
        action = action.group(1)
        start = text.find(f"ACTION: {action}")
        for probing_length in range(15, 160, 10):
            selected_option_from_action = re.findall(
                r"ELEMENT: ([A-Z]{2}|[A-Z])",
                text[max(start - probing_length, 0):start])
            # print("text span:",text[max(start-probing_length,0):start])
            # print("finded group:",selected_option__)
            if selected_option_from_action and len(selected_option_from_action) > 0:
                selected_option = selected_option_from_action[0]
                break
    else:
        action = "None"

    selected_option = re.search(r"ELEMENT: (.*)$", text, re.MULTILINE)
    selected_option = selected_option.group(1) if selected_option is not None else ""

    value = re.search(r"VALUE: (.*)$", text, re.MULTILINE)
    value = value.group(1) if value is not None else ""
    return selected_option, (action or "").strip(), process_string(process_string((value or "").strip()))

def process_string(input_string):
    if input_string.startswith('"') and input_string.endswith('"'):
        input_string = input_string[1:-1]
    if input_string.endswith('.'):
        input_string = input_string[:-1]
    return input_string







