"""
DOM-based operations for faster and more reliable SELECT and TYPE actions.
Uses Playwright to extract all form elements and operate on them directly without coordinates.

This module provides:
1. Element extraction: get_all_input_elements(), get_all_select_elements()
2. LLM decision making: decide_form_element() to choose which element to use
3. Direct DOM operations: dom_type_by_selector(), dom_select_by_selector()
"""

from typing import List, Dict, Optional, Tuple
import logging
import re
import sys


# Removed legacy input extraction used by old TYPE flow


async def get_all_select_elements(page) -> List[Dict]:
    """
    Extract all select dropdown elements from the page using Playwright.
    
    Returns:
        List of select element info dicts:
        [
            {
                'index': 0,
                'id': 'country',
                'name': 'country',
                'label': 'Country',
                'description': 'Country dropdown with 195 options',
                'selector': 'select#country',
                'current_value': 'us',
                'current_text': 'United States',
                'options': [
                    {'value': 'us', 'text': 'United States', 'index': 0},
                    {'value': 'ca', 'text': 'Canada', 'index': 1},
                    ...
                ],
                'visible': True
            },
            ...
        ]
    """
    try:
        elements = await page.evaluate("""
            () => {
                const selects = [];
                const allSelects = document.querySelectorAll('select');
                
                allSelects.forEach((el, index) => {
                    // Check if visible
                    const isVisible = el.offsetParent !== null && 
                                     window.getComputedStyle(el).display !== 'none' &&
                                     window.getComputedStyle(el).visibility !== 'hidden';
                    
                    if (!isVisible) return;
                    
                    // Generate unique selector
                    let selector = '';
                    if (el.id) {
                        selector = `select[id="${el.id}"]`;
                    } else if (el.name) {
                        selector = `select[name="${el.name}"]`;
                    } else {
                        selector = `select:nth-of-type(${selects.length + 1})`;
                    }
                    
                    // Get all options
                    const options = Array.from(el.options).map((opt, idx) => ({
                        value: opt.value,
                        text: opt.text.trim(),
                        index: idx,
                        selected: opt.selected
                    }));
                    
                    // Build description
                    const label = el.labels?.[0]?.textContent?.trim() || 
                                 document.querySelector(`label[for="${el.id}"]`)?.textContent?.trim();
                    
                    let description = '';
                    if (label) {
                        description = `${label} dropdown with ${options.length} options`;
                    } else if (el.name) {
                        description = `${el.name} dropdown with ${options.length} options`;
                    } else {
                        description = `Dropdown with ${options.length} options`;
                    }
                    
                    const selectedOption = el.options[el.selectedIndex];
                    
                    selects.push({
                        index: selects.length,
                        id: el.id || '',
                        name: el.name || '',
                        label: label || '',
                        description: description,
                        selector: selector,
                        current_value: selectedOption ? selectedOption.value : '',
                        current_text: selectedOption ? selectedOption.text.trim() : '',
                        options: options,
                        visible: isVisible,
                        disabled: el.disabled
                    });
                });
                
                return selects;
            }
        """)
        
        return elements
        
    except Exception as e:
        logging.error(f"Failed to get select elements: {e}")
        return []


async def get_select_options(page, coordinates: Tuple[int, int]) -> Optional[Dict]:
    """
    Get all available options from a select dropdown at given coordinates.
    
    Args:
        page: Playwright page object
        coordinates: (x, y) tuple of the select element location
        
    Returns:
        Dict with select element info and available options, or None if not found
        {
            'selector': 'select#country',
            'current_value': 'us',
            'current_text': 'United States',
            'options': [
                {'value': 'us', 'text': 'United States', 'index': 0},
                {'value': 'ca', 'text': 'Canada', 'index': 1},
                ...
            ]
        }
    """
    try:
        x, y = coordinates
        
        # Find the select element at these coordinates using Playwright DOM API
        select_element = await page.evaluate(f"""
            () => {{
                const element = document.elementFromPoint({x}, {y});
                if (!element) return null;
                
                // Find closest select element (might be clicking on container)
                let selectEl = element;
                if (selectEl.tagName !== 'SELECT') {{
                    selectEl = element.closest('select');
                }}
                if (!selectEl) return null;
                
                // Get all options
                const options = Array.from(selectEl.options).map((opt, index) => ({{
                    value: opt.value,
                    text: opt.text.trim(),
                    index: index,
                    selected: opt.selected
                }}));
                
                // Get current selection
                const selectedOption = selectEl.options[selectEl.selectedIndex];
                
                // Generate a unique selector for this element
                let selector = selectEl.id ? `select[id="${selectEl.id}"]` : null;
                if (!selector && selectEl.name) {{
                    selector = `select[name="${{selectEl.name}}"]`;
                }}
                if (!selector) {{
                    // Fallback: use position in DOM
                    const selects = Array.from(document.querySelectorAll('select'));
                    const index = selects.indexOf(selectEl);
                    selector = `select:nth-of-type(${{index + 1}})`;
                }}
                
                return {{
                    selector: selector,
                    current_value: selectedOption ? selectedOption.value : '',
                    current_text: selectedOption ? selectedOption.text.trim() : '',
                    options: options,
                    tag_name: 'select'
                }};
            }}
        """)
        
        return select_element
        
    except Exception as e:
        logging.error(f"Failed to get select options: {e}")
        return None


async def dom_select_option(page, coordinates: Tuple[int, int], option_text: str) -> Tuple[bool, str]:
    """
    Select an option from a dropdown using DOM manipulation (no scrolling needed).
    
    Args:
        page: Playwright page object
        coordinates: (x, y) tuple of the select element
        option_text: Text of the option to select
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # First, get available options
        select_info = await get_select_options(page, coordinates)
        
        if not select_info:
            return False, "Could not find select element at coordinates"
        
        selector = select_info['selector']
        options = select_info['options']
        
        # Find matching option (try multiple strategies)
        matched_option = None
        option_text_lower = option_text.lower().strip()
        
        # Strategy 1: Exact text match
        for opt in options:
            if opt['text'].lower() == option_text_lower:
                matched_option = opt
                break
        
        # Strategy 2: Partial text match
        if not matched_option:
            for opt in options:
                if option_text_lower in opt['text'].lower():
                    matched_option = opt
                    break
        
        # Strategy 3: Value match
        if not matched_option:
            for opt in options:
                if opt['value'].lower() == option_text_lower:
                    matched_option = opt
                    break
        
        if not matched_option:
            available_options = [opt['text'] for opt in options]
            return False, f"Option '{option_text}' not found. Available: {available_options}"
        
        # Select the option using DOM (bypasses scrolling)
        success = await page.evaluate("""
            ([selector, index]) => {
                const selectEl = document.querySelector(selector);
                if (!selectEl) return false;
                
                selectEl.selectedIndex = index;
                
                // Trigger change events for JavaScript listeners
                selectEl.dispatchEvent(new Event('change', { bubbles: true }));
                selectEl.dispatchEvent(new Event('input', { bubbles: true }));
                
                return true;
            }
        """, [selector, matched_option['index']])
        
        if success:
            return True, f"Selected '{matched_option['text']}' from dropdown"
        else:
            return False, "Failed to select option via DOM"
            
    except Exception as e:
        logging.error(f"DOM select failed: {e}")
        return False, f"Error: {str(e)}"


# Removed legacy coordinate-to-input resolver used by old TYPE flow


# Removed legacy DOM typing used by old TYPE flow


# Removed legacy selector-based typing used by old TYPE flow


async def dom_select_by_selector(page, selector: str, option_text: str) -> Tuple[bool, str]:
    """
    Select an option from dropdown by selector (no coordinates needed).
    
    Args:
        page: Playwright page object
        selector: CSS selector for the select element
        option_text: Text of the option to select
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Get options and select
        result = await page.evaluate("""
            ([selector, optionText]) => {
                const selectEl = document.querySelector(selector);
                if (!selectEl) return { success: false, message: 'Element not found' };
                
                if (selectEl.disabled) {
                    return { success: false, message: 'Element is disabled' };
                }
                
                // Get all options
                const options = Array.from(selectEl.options);
                const optionTextLower = optionText.toLowerCase().trim();
                
                // Find matching option (try multiple strategies)
                let matchedOption = null;
                
                // Strategy 1: Exact text match
                matchedOption = options.find(opt => opt.text.toLowerCase() === optionTextLower);
                
                // Strategy 2: Partial text match
                if (!matchedOption) {
                    matchedOption = options.find(opt => opt.text.toLowerCase().includes(optionTextLower));
                }
                
                // Strategy 3: Value match
                if (!matchedOption) {
                    matchedOption = options.find(opt => opt.value.toLowerCase() === optionTextLower);
                }
                
                if (!matchedOption) {
                    const available = options.map(opt => opt.text).join(', ');
                    return { 
                        success: false, 
                        message: `Option '${optionText}' not found. Available: ${available}` 
                    };
                }
                
                // Select the option
                selectEl.value = matchedOption.value;
                selectEl.selectedIndex = options.indexOf(matchedOption);
                
                // Trigger change events
                selectEl.dispatchEvent(new Event('change', { bubbles: true }));
                selectEl.dispatchEvent(new Event('input', { bubbles: true }));
                
                return { 
                    success: true, 
                    message: `Selected '${matchedOption.text}' from dropdown` 
                };
            }
        """, [selector, option_text])
        
        return result['success'], result['message']
        
    except Exception as e:
        logging.error(f"DOM select by selector failed: {e}")
        return False, f"Error: {str(e)}"


# Removed legacy listing util relying on old TYPE extraction


# ============================================================================
# SMART MATCHING (Try exact match first, avoid LLM if possible)
# ============================================================================

# ============================================================================
# LLM-BASED DECISION MAKING
# ============================================================================

def build_form_elements_prompt(inputs: List[Dict], selects: List[Dict], action_type: str, value: str, task: str) -> str:
    """
    Build a prompt for the LLM to decide which form element to interact with.
    
    Args:
        inputs: List of available input elements
        selects: List of available select elements
        action_type: Either "TYPE" or "SELECT"
        value: The value to type or select
        task: The current task description
        
    Returns:
        Formatted prompt string
    """
    if action_type == "TYPE":
        elements_list = "\n".join([
            f"  [{i}] {inp['description']}"
            f" (type={inp['type']}, current_value='{inp['value']}')"
            for i, inp in enumerate(inputs)
        ])
        
        prompt = f"""You need to TYPE the text "{value}" into an input field.

**Current Task**: {task}

**Available Input Fields**:
{elements_list if elements_list else "  No input fields found"}

**Your Decision**:
Which input field should receive this text? Respond with ONLY the number in brackets [N].
If none of the fields are appropriate, respond with "NONE".

Example response: "0" or "2" or "NONE"
"""
    
    else:  # SELECT
        elements_list = "\n".join([
            f"  [{i}] {sel['description']}"
            f" (current='{sel['current_text']}', options={len(sel['options'])})"
            for i, sel in enumerate(selects)
        ])
        
        prompt = f"""You need to SELECT the option "{value}" from a dropdown.

**Current Task**: {task}

**Available Dropdowns**:
{elements_list if elements_list else "  No dropdown fields found"}

**Your Decision**:
Which dropdown should be used for selecting "{value}"? Respond with ONLY the number in brackets [N].
If none of the dropdowns are appropriate, respond with "NONE".

Example response: "0" or "1" or "NONE"
"""
    
    return prompt


async def decide_form_element(engine, inputs: List[Dict], selects: List[Dict], 
                              action_type: str, value: str, task: str, logger=None) -> Optional[Dict]:
    """
    Use LLM to decide which form element to interact with.
    
    Args:
        engine: LLM inference engine
        inputs: List of available input elements
        selects: List of available select elements
        action_type: Either "TYPE" or "SELECT"
        value: The value to type or select
        task: The current task description
        logger: Optional logger instance
        
    Returns:
        Selected element dict or None if no appropriate element found
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Build prompt
    prompt = build_form_elements_prompt(inputs, selects, action_type, value, task)
    
    try:
        # Call LLM
        logger.info(f"🤔 Asking LLM to decide which element for {action_type} action...")
        response = await engine.generate(
            prompt=[prompt, "", ""],
            image_path=None,  # No screenshot needed for element selection
            turn_number=0
        )
        
        # Handle list response
        if isinstance(response, list):
            response = response[0] if response else ""
        
        logger.info(f"LLM decision: {response}")
        
        # Parse response
        response = response.strip()
        
        if response.upper() == "NONE" or not response:
            logger.warning(f"LLM said NONE - no appropriate element found")
            return None
        
        # Extract number
        try:
            index = int(response)
        except ValueError:
            # Try to extract number from response
            numbers = re.findall(r'\d+', response)
            if numbers:
                index = int(numbers[0])
            else:
                logger.error(f"Could not parse index from LLM response: {response}")
                return None
        
        # Get the element
        if action_type == "TYPE":
            if 0 <= index < len(inputs):
                selected = inputs[index]
                logger.info(f"✅ Selected input [{index}]: {selected['description']}")
                return selected
            else:
                logger.error(f"Index {index} out of range for inputs (0-{len(inputs)-1})")
                return None
        else:  # SELECT
            if 0 <= index < len(selects):
                selected = selects[index]
                logger.info(f"✅ Selected dropdown [{index}]: {selected['description']}")
                return selected
            else:
                logger.error(f"Index {index} out of range for selects (0-{len(selects)-1})")
                return None
        
    except Exception as e:
        logger.error(f"Error in LLM form element decision: {e}")
        return None


async def extract_typeable_elements(page, logger=None):
    try:
        elements = await page.evaluate("""
            () => {
                const elements = [];
                const selectors = [
                    'input[type="text"]',
                    'input[type="search"]',
                    'input[type="email"]',
                    'input[type="tel"]',
                    'input[type="url"]',
                    'input[type="password"]',
                    'input[type="number"]',
                    'input:not([type])',
                    'textarea',
                    '[contenteditable="true"]',
                    '[role="textbox"]'
                ];
                
                selectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach((el, idx) => {
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            let label = '';
                            if (el.id) {
                                const labelEl = document.querySelector(`label[for="${el.id}"]`);
                                if (labelEl) label = labelEl.textContent.trim();
                            }
                            if (!label && el.labels && el.labels.length > 0) {
                                label = el.labels[0].textContent.trim();
                            }
                            if (!label) {
                                label = el.getAttribute('aria-label') || el.getAttribute('placeholder') || '';
                            }
                            
                            elements.push({
                                index: elements.length,
                                tagName: el.tagName.toLowerCase(),
                                type: el.type || 'text',
                                id: el.id || '',
                                name: el.name || '',
                                placeholder: el.placeholder || '',
                                label: label,
                                value: el.value || el.textContent || '',
                                center: {
                                    x: rect.left + rect.width / 2,
                                    y: rect.top + rect.height / 2
                                },
                                selector: el.id ? `${el.tagName.toLowerCase()}[id="${el.id}"]` : 
                                         el.name ? `${el.tagName.toLowerCase()}[name="${el.name}"]` :
                                         el.className ? `${el.tagName.toLowerCase()}.${el.className.split(' ')[0]}` :
                                         `${selector}:nth-of-type(${idx + 1})`
                            });
                        }
                    });
                });
                
                return elements;
            }
        """)
        return elements
    except Exception as e:
        if logger:
            logger.error(f"Failed to extract typeable elements: {e}")
        else:
            logging.error(f"Failed to extract typeable elements: {e}")
        return []


async def verify_last_typing(page, expected_value: str) -> bool:
    try:
        val = await page.evaluate("() => { const el = document.activeElement; if (!el) return ''; const tag = (el.tagName || '').toLowerCase(); if (tag === 'input' || tag === 'textarea') return el.value || ''; if (el.isContentEditable) return el.textContent || ''; return ''; }")
        if not isinstance(val, str):
            return False
        ev = (expected_value or '').strip()
        if not ev:
            return False
        return ev in (val or '')
    except Exception:
        return False


async def clear_active_field(page):
    try:
        if sys.platform == "darwin":
            try:
                await page.keyboard.press("Meta+A")
            except Exception:
                pass
        else:
            try:
                await page.keyboard.press("Control+A")
            except Exception:
                pass
        try:
            await page.keyboard.press("Backspace")
        except Exception:
            pass
    except Exception:
        return


def looks_like_api_endpoint(url: str) -> bool:
    try:
        if not url:
            return False
        u = url.lower()
        bad = ["/graphql", "/api/", "/api", "/v1/", "/v2/", "/v3/", "/rest/", ".json", ".csv", ".xml", "?format=json", "output=json"]
        for p in bad:
            if p in u:
                return True
        return False
    except Exception:
        return False


async def extract_selectable_elements(page, logger=None):
    try:
        elements = await page.evaluate("""
            () => {
                const elements = [];
                document.querySelectorAll('select').forEach((el, idx) => {
                    const rect = el.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        let label = '';
                        if (el.id) {
                            const labelEl = document.querySelector(`label[for="${el.id}"]`);
                            if (labelEl) label = labelEl.textContent.trim();
                        }
                        if (!label && el.labels && el.labels.length > 0) {
                            label = el.labels[0].textContent.trim();
                        }
                        
                        const options = [];
                        el.querySelectorAll('option').forEach(opt => {
                            if (opt.value || opt.textContent.trim()) {
                                options.push({
                                    value: opt.value,
                                    text: opt.textContent.trim()
                                });
                            }
                        });
                        
                        elements.push({
                            index: elements.length,
                            id: el.id || '',
                            name: el.name || '',
                            label: label,
                            currentValue: el.value || '',
                            options: options,
                            center: {
                                x: rect.left + rect.width / 2,
                                y: rect.top + rect.height / 2
                            },
                            selector: el.id ? `select[id="${el.id}"]` : 
                                     el.name ? `select[name="${el.name}"]` :
                                     `select:nth-of-type(${idx + 1})`
                        });
                    }
                });
                
                return elements;
            }
        """)
        return elements
    except Exception as e:
        if logger:
            logger.error(f"Failed to extract selectable elements: {e}")
        else:
            logging.error(f"Failed to extract selectable elements: {e}")
        return []


async def choose_field_with_llm(engine, model, logger, elements, action_type, intended_value, task):
    if not elements:
        return None
    if action_type == "TYPE":
        elements_desc = "\n".join([
            f"{i}. {el.get('label') or el.get('placeholder') or el.get('name') or el.get('id') or 'Unnamed field'} "
            f"(type: {el.get('type')}, current: '{el.get('value', '')}')"
            for i, el in enumerate(elements)
        ])
        prompt = f"""You need to TYPE "{intended_value}" for the task: "{task}"

Available input fields:
{elements_desc}

Which field should be used? Respond with ONLY the number (0-{len(elements)-1}).
"""
    else:
        elements_desc = "\n".join([
            f"{i}. {el.get('label') or el.get('name') or el.get('id') or 'Unnamed select'} "
            f"(current: '{el.get('currentValue')}', options: {len(el.get('options', []))} available)"
            for i, el in enumerate(elements)
        ])
        prompt = f"""You need to SELECT "{intended_value}" for the task: "{task}"

Available dropdowns:
{elements_desc}

Which dropdown should be used? Respond with ONLY the number (0-{len(elements)-1}).
"""
    
    try:
        response = await engine.generate(
            prompt=[prompt, "", ""],
            image_path=None,
            temperature=0.0,
            model=model,
            turn_number=0
        )
        match = re.search(r'\b(\d+)\b', response)
        if match:
            index = int(match.group(1))
            if 0 <= index < len(elements):
                if logger:
                    logger.info(f"LLM chose field {index}: {elements[index]}")
                return elements[index]
        if logger:
            logger.warning(f"Could not parse LLM response '{response}', using first element")
        return elements[0]
    except Exception as e:
        if logger:
            logger.error(f"Failed to choose field with LLM: {e}")
        return elements[0] if elements else None
