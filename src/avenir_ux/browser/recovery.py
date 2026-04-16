import asyncio
import os
import json
import re
import litellm
import time
import random
import traceback
from difflib import SequenceMatcher
from ..utils.image import encode_image_with_compression
from ..llm.engine import add_llm_io_record


def extract_target_from_action_generation(action_generation, logger=None):
    """
    Extract target element description from action generation output.
    Enhanced version with better element identification and context awareness.
    """
    try:
        # Check if this is a SAY command (thinking step, not task completion)
        if action_generation and action_generation.strip().upper().startswith('SAY:'):
            if logger:
                logger.info("Detected SAY command - thinking step, not extracting target")
            return None
        
        # Look for common patterns in action generation
        lines = action_generation.split('\n')
        
        # Enhanced target extraction with priority keywords
        priority_keywords = ['play simulation', 'start simulation', 'run simulation', 'simulation button']
        secondary_keywords = ['click', 'select', 'choose', 'find', 'locate', 'press', 'tap']
        
        # First pass: look for high-priority simulation-related targets
        for line in lines:
            line_lower = (line or "").strip().lower()
            for priority_keyword in priority_keywords:
                if priority_keyword in line_lower:
                    if logger:
                        logger.info(f"Found priority target: {priority_keyword}")
                    return priority_keyword
        
        # Second pass: extract targets from action descriptions
        for line in lines:
            line_lower = (line or "").strip().lower()
            
            # Look for lines that describe what to click/interact with
            for keyword in secondary_keywords:
                if keyword in line_lower:
                    # Extract the target description after the keyword
                    parts = line_lower.split(keyword)
                    if len(parts) > 1:
                        target = (parts[1] or "").strip()
                        # Clean up the target description
                        target = target.replace('on', '').replace('the', '').replace('button', '').strip()
                        # Remove common prefixes and suffixes
                        target = target.strip('.,!?;:')
                        
                        # Filter out unwanted targets
                        unwanted_targets = ['google classroom', 'add to google', 'login', 'sign in', 'register']
                        if target and not any(unwanted in target for unwanted in unwanted_targets):
                            if logger:
                                logger.info(f"Extracted target from '{keyword}': {target}")
                            return target
        
        # Third pass: look for element descriptions in quotes or specific formats
        import re
        element_pattern = r'element[:\s]*["\']([^"\']+)["\']'
        match = re.search(element_pattern, action_generation, re.IGNORECASE)
        if match:
            target = match.group(1).strip()
            if logger:
                logger.info(f"Extracted target from regex: {target}")
            return target

        if logger:
            logger.warning("Could not extract a clear target from action generation")
        return None
    except Exception as e:
        if logger:
            logger.error(f"Error extracting target from action generation: {e}")
        return None


async def attempt_scroll_recovery(page, scroll_height, action_generation_output, logger=None):
    """
    Attempt recovery by scrolling and re-grounding.
    Always returns a valid result, never None.
    """
    if logger:
        logger.info("Attempting scroll recovery...")
    
    scroll_distance = scroll_height // 2
    
    # Try scrolling down first
    await page.evaluate(f"window.scrollBy(0, {scroll_distance});")
    await asyncio.sleep(1)
    
    if logger:
        logger.info("Scroll recovery executed")
    return {
        "action_generation": action_generation_output,
        "action_grounding": "Scroll down to find target element",
        "element": None,
        "action": "SCROLL DOWN",
        "value": "None",
        "recovery_method": "scroll_recovery"
    }


async def attempt_gui_grounding_recovery(
    page,
    screenshot_path,
    action_generation_output,
    gui_grounding_model,
    gui_grounding_base_url,
    gui_grounding_api_key,
    viewport_width,
    viewport_height,
    target_instruction=None,
    logger=None,
):
    """
    Attempt recovery using GUI grounding API when target element is believed to be on page.
    Always returns a valid result or empty dict, never None.
    """
    if logger:
        logger.info("Attempting GUI grounding recovery...")
    
    # Use provided target_instruction or extract from action generation
    if target_instruction is None:
        target_instruction = extract_target_from_action_generation(action_generation_output, logger)
    
    if not target_instruction:
        if logger:
            logger.warning("Could not extract target instruction for GUI grounding")
        return {}
    
    # Use GUI grounding to locate element
    coords = None
    if os.path.exists(screenshot_path):
        coords = await ground_element(
            instruction=target_instruction,
            image_path=screenshot_path,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            gui_grounding_model=gui_grounding_model,
            gui_grounding_base_url=gui_grounding_base_url,
            gui_grounding_api_key=gui_grounding_api_key,
            logger=logger,
        )
    else:
        if logger:
            logger.warning(f"Screenshot not found for GUI grounding: {screenshot_path}")
    
    if not coords:
        if logger:
            logger.info("GUI grounding could not locate target element")
        return {}
    
    if logger:
        logger.info(f"GUI grounding found target at coordinates: {coords}")
    
    # Create a synthetic element for the found coordinates
    synthetic_element = {
        "center_point": coords,
        "description": f"GUI grounded element: {target_instruction}",
        "tag_with_role": "gui_grounded",
        "tag": "gui_grounded",
        "selector": "coordinate_based",
        "box": {
            "x": coords[0] - 10,
            "y": coords[1] - 10,
            "width": 20,
            "height": 20
        },
        "gui_grounded": True,
        "coordinates": coords,
        "clickable": True
    }
    
    return {
        "action_generation": action_generation_output,
        "action_grounding": f"GUI grounding located target at {coords}",
        "element": synthetic_element,
        "action": "CLICK",
        "value": "",
        "recovery_method": "gui_grounding"
    }


def encode_image_for_grounding(image_path):
    return encode_image_with_compression(image_path)


def parse_grounding_coordinates(text, map_normalized_to_pixels, logger=None):
    try:
        if text.strip().startswith('{') and text.strip().endswith('}'):
            data = json.loads(text.strip())
            if 'arguments' in data and 'coordinate' in data['arguments']:
                coords = data['arguments']['coordinate']
                if isinstance(coords, list) and len(coords) == 2:
                    x, y = int(coords[0]), int(coords[1])
                    if logger:
                        logger.info(f"Parsed JSON coordinates: ({x}, {y})")
                    return map_normalized_to_pixels(x, y)
        else:
            json_match = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}', text)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                if 'arguments' in data and 'coordinate' in data['arguments']:
                    coords = data['arguments']['coordinate']
                    if isinstance(coords, list) and len(coords) == 2:
                        x, y = int(coords[0]), int(coords[1])
                        if logger:
                            logger.info(f"Parsed JSON coordinates: ({x}, {y})")
                        return map_normalized_to_pixels(x, y)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        if logger:
            logger.debug(f"Failed to parse JSON format: {e}")
    
    match = re.search(r'<\|box_start\|>\((\d+),\s*(\d+)\)<\|box_end\|>', text)
    if not match:
        match = re.search(r'\((\d+),\s*(\d+)\)', text)
    
    if match:
        x, y = int(match.group(1)), int(match.group(2))
        return map_normalized_to_pixels(x, y)
    return None


async def run_gui_grounding(image_path, instruction, model, map_normalized_to_pixels, logger=None):
    try:
        if not image_path or not os.path.exists(image_path):
            if logger:
                logger.error(f"Invalid image path for GUI grounding: {image_path}")
            return None
        if not instruction or not instruction.strip():
            if logger:
                logger.error("Empty instruction for GUI grounding")
            return None
        
        base64_image = encode_image_for_grounding(image_path)
        system_content = (
            "You are a GUI grounding model. Given a screenshot and an instruction, "
            "output coordinates in the format: <|box_start|>(x,y)<|box_end|> where x and y are integers in [0,1000]. "
            "Do NOT output any explanation or extra text."
        )
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 100,
            "temperature": 0.1
        }
        
        if logger:
            logger.info(f"Sending GUI grounding request for: {instruction}")
        
        response = await litellm.acompletion(
            model=model,
            messages=payload["messages"],
            max_tokens=100,
            temperature=0.1,
            api_key=os.getenv('OPENROUTER_API_KEY')
        )
        
        if response and response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if choice.message and choice.message.content:
                content = choice.message.content.strip()
                if logger:
                    logger.info(f"Coordinate response: {content}")
                try:
                    add_llm_io_record({
                        "model": model if isinstance(model, str) else str(model),
                        "turn_number": 0,
                        "messages": payload["messages"],
                        "image_paths": [image_path] if image_path else None,
                        "output": content,
                        "context": "coordinate_extraction"
                    })
                except Exception:
                    pass
                coords = parse_grounding_coordinates(content, map_normalized_to_pixels, logger=logger)
                if coords and logger:
                    logger.info(f"Successfully parsed coordinates: {coords}")
                if not coords and logger:
                    logger.warning(f"Failed to parse coordinates from response: {content}")
                return coords
            if logger:
                logger.error("Invalid message structure in coordinate response")
            return None
        if logger:
            logger.error("Invalid response format from coordinate API")
        return None
    except Exception as e:
        if logger:
            logger.error(f"Coordinate request failed: {e}")
        return None

async def capture_page_state(page, logger=None, pending_hit_test_coords=None):
    try:
        state = {
            'url': '',
            'title': '',
            'visible_text': '',
            'interactive_elements_count': 0,
            'form_fields_count': 0,
            'modal_present': False,
            'scroll_x': 0,
            'scroll_y': 0,
            'active_element': None
        }
        
        if page and page.is_closed():
            state['error'] = 'page_closed'
            return state
        
        try:
            await page.wait_for_load_state('domcontentloaded', timeout=10000)
        except Exception:
            pass
        
        try:
            state['url'] = page.url
        except Exception:
            state['url'] = ''
        
        try:
            state['title'] = await page.title()
        except Exception:
            state['title'] = ''
        
        visible_text = await page.evaluate("""
            () => {
                const root = document.body || document.documentElement;
                if (!root) { return ''; }
                let walker;
                try {
                    walker = document.createTreeWalker(
                        root,
                        NodeFilter.SHOW_TEXT,
                        {
                            acceptNode: function(node) {
                                const parent = node.parentElement;
                                if (!parent) return NodeFilter.FILTER_REJECT;
                                
                                const style = window.getComputedStyle(parent);
                                if (style.display === 'none' || 
                                    style.visibility === 'hidden' || 
                                    style.opacity === '0') {
                                    return NodeFilter.FILTER_REJECT;
                                }
                                
                                return NodeFilter.FILTER_ACCEPT;
                            }
                        }
                    );
                } catch (e) {
                    return '';
                }
                
                let text = '';
                let node;
                while (node = walker.nextNode()) {
                    text += node.textContent.trim() + ' ';
                    if (text.length > 1000) break;
                }
                return text.trim();
            }
        """)
        state['visible_text'] = visible_text[:1000] if visible_text else ""
        
        interactive_count = await page.evaluate("""
            () => {
                const interactiveElements = document.querySelectorAll(
                    'button, input, select, textarea, a[href], [onclick], [role="button"]'
                );
                return interactiveElements.length;
            }
        """)
        state['interactive_elements_count'] = interactive_count
        
        form_fields_count = await page.evaluate("""
            () => {
                const formFields = document.querySelectorAll('input, select, textarea');
                return formFields.length;
            }
        """)
        state['form_fields_count'] = form_fields_count
        
        modal_present = await page.evaluate("""
            () => {
                const modals = document.querySelectorAll(
                    '[role="dialog"], .modal, .popup, .overlay, [aria-modal="true"]'
                );
                return modals.length > 0;
            }
        """)
        state['modal_present'] = modal_present

        focus_state = await page.evaluate("""
            () => {
                const sx = window.scrollX || 0;
                const sy = window.scrollY || 0;
                const ae = document.activeElement;
                let active = null;
                try {
                    if (ae && ae !== document.body) {
                        const tag = (ae.tagName || '').toUpperCase();
                        const val = (typeof ae.value === 'string') ? ae.value : '';
                        let selectedText = '';
                        let selectedIndex = null;
                        if (tag === 'SELECT') {
                            selectedIndex = typeof ae.selectedIndex === 'number' ? ae.selectedIndex : null;
                            try {
                                const opt = ae.selectedOptions && ae.selectedOptions.length ? ae.selectedOptions[0] : null;
                                selectedText = opt ? (opt.textContent || '') : '';
                            } catch (e) {
                                selectedText = '';
                            }
                        }
                        active = {
                            tag,
                            id: ae.id || '',
                            name: ae.getAttribute && ae.getAttribute('name') ? ae.getAttribute('name') : '',
                            type: ae.getAttribute && ae.getAttribute('type') ? ae.getAttribute('type') : '',
                            role: ae.getAttribute && ae.getAttribute('role') ? ae.getAttribute('role') : '',
                            placeholder: ae.getAttribute && ae.getAttribute('placeholder') ? ae.getAttribute('placeholder') : '',
                            ariaLabel: ae.getAttribute && ae.getAttribute('aria-label') ? ae.getAttribute('aria-label') : '',
                            value: (val || '').slice(0, 200),
                            selectedText: (selectedText || '').slice(0, 200),
                            selectedIndex
                        };
                    }
                } catch (e) {
                    active = null;
                }
                return { sx, sy, active };
            }
        """)
        if isinstance(focus_state, dict):
            try:
                state['scroll_x'] = int(focus_state.get('sx', 0) or 0)
            except Exception:
                state['scroll_x'] = 0
            try:
                state['scroll_y'] = int(focus_state.get('sy', 0) or 0)
            except Exception:
                state['scroll_y'] = 0
            state['active_element'] = focus_state.get('active')

        coords = pending_hit_test_coords
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            try:
                hx, hy = int(coords[0]), int(coords[1])
                hit_state = await page.evaluate(
                    """
                    ([x, y]) => {
                        try {
                            const el = document.elementFromPoint(x, y);
                            if (!el) return null;
                            const r = el.getBoundingClientRect();
                            const tag = (el.tagName || '').toUpperCase();
                            const txt = (el.innerText || el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 200);
                            const ariaLabel = (el.getAttribute && el.getAttribute('aria-label')) ? el.getAttribute('aria-label') : '';
                            const role = (el.getAttribute && el.getAttribute('role')) ? el.getAttribute('role') : '';
                            const href = (el.getAttribute && el.getAttribute('href')) ? el.getAttribute('href') : '';
                            const disabled = (typeof el.disabled === 'boolean') ? el.disabled : (el.getAttribute && el.getAttribute('aria-disabled') === 'true');
                            const ariaPressed = (el.getAttribute && el.getAttribute('aria-pressed')) ? el.getAttribute('aria-pressed') : '';
                            const ariaExpanded = (el.getAttribute && el.getAttribute('aria-expanded')) ? el.getAttribute('aria-expanded') : '';
                            const type = (el.getAttribute && el.getAttribute('type')) ? el.getAttribute('type') : '';
                            const value = (typeof el.value === 'string') ? el.value.slice(0, 200) : '';
                            return {
                                x,
                                y,
                                tag,
                                id: el.id || '',
                                name: (el.getAttribute && el.getAttribute('name')) ? el.getAttribute('name') : '',
                                className: (el.className && typeof el.className === 'string') ? el.className.slice(0, 200) : '',
                                role,
                                ariaLabel,
                                href,
                                disabled,
                                ariaPressed,
                                ariaExpanded,
                                type,
                                value,
                                text: txt,
                                rect: { left: r.left, top: r.top, width: r.width, height: r.height }
                            };
                        } catch (e) {
                            return null;
                        }
                    }
                    """,
                    [hx, hy]
                )
                state['hit_target'] = hit_state
            except Exception:
                state['hit_target'] = None
        
        return state
        
    except Exception as e:
        if logger:
            try:
                logger.warning(f"Failed to capture page state: {e}")
            except Exception:
                pass
        return {
            'error': str(e),
            'url': '',
            'title': '',
            'visible_text': '',
            'interactive_elements_count': 0,
            'form_fields_count': 0,
            'modal_present': False
        }


def detect_page_state_change(state_before, state_after, action_type, logger=None, state_store=None):
    try:
        if not isinstance(state_before, dict):
            state_before = {}
        if not isinstance(state_after, dict):
            state_after = {}
        if state_before.get('error') or state_after.get('error'):
            return True
        
        changes_detected = []
        
        vb = state_before.get('visible_text')
        va = state_after.get('visible_text')
        text_before = vb if isinstance(vb, str) else ''
        text_after = va if isinstance(va, str) else ''
        if len(text_before) > 0 and len(text_after) > 0:
            similarity = SequenceMatcher(None, text_before, text_after).ratio()
            if similarity < 0.95:
                changes_detected.append('content_changed')
                if logger:
                    logger.info(f"Content changed significantly (similarity: {similarity:.2f})")
        
        ib = state_before.get('interactive_elements_count')
        ia = state_after.get('interactive_elements_count')
        try:
            ib = int(ib) if ib is not None else 0
        except Exception:
            ib = 0
        try:
            ia = int(ia) if ia is not None else 0
        except Exception:
            ia = 0
        if ib != ia:
            changes_detected.append('interactive_elements_changed')
            if logger:
                logger.info(f"Interactive elements count changed: {ib} -> {ia}")
        
        fb = state_before.get('form_fields_count')
        fa = state_after.get('form_fields_count')
        try:
            fb = int(fb) if fb is not None else 0
        except Exception:
            fb = 0
        try:
            fa = int(fa) if fa is not None else 0
        except Exception:
            fa = 0
        if fb != fa:
            changes_detected.append('form_fields_changed')
            if logger:
                logger.info(f"Form fields count changed: {fb} -> {fa}")
        
        tb = state_before.get('title') if isinstance(state_before.get('title'), str) else ''
        ta = state_after.get('title') if isinstance(state_after.get('title'), str) else ''
        if tb != ta:
            changes_detected.append('title_changed')
            if logger:
                logger.info(f"Title changed: {tb} -> {ta}")
        
        ub = state_before.get('url') if isinstance(state_before.get('url'), str) else ''
        ua = state_after.get('url') if isinstance(state_after.get('url'), str) else ''
        if ub != ua:
            changes_detected.append('url_changed')
            if logger:
                logger.info(f"URL changed: {ub} -> {ua} (Note: URL change is secondary to content changes)")
        
        mpb = bool(state_before.get('modal_present', False))
        mpa = bool(state_after.get('modal_present', False))
        if mpb != mpa:
            changes_detected.append('modal_state_changed')
            if logger:
                logger.info(f"Modal/popup state changed: {mpb} -> {mpa}")

        try:
            sbx = int(state_before.get('scroll_x', 0) or 0)
            sby = int(state_before.get('scroll_y', 0) or 0)
            sax = int(state_after.get('scroll_x', 0) or 0)
            say = int(state_after.get('scroll_y', 0) or 0)
            if abs(sax - sbx) >= 5 or abs(say - sby) >= 5:
                changes_detected.append('scroll_changed')
                if logger:
                    logger.info(f"Scroll changed: ({sbx},{sby}) -> ({sax},{say})")
        except Exception:
            pass

        ab = state_before.get('active_element')
        aa = state_after.get('active_element')
        try:
            if isinstance(ab, dict) and isinstance(aa, dict):
                focus_keys = ('tag', 'id', 'name', 'type', 'role', 'placeholder', 'ariaLabel')
                if any((ab.get(k) or '') != (aa.get(k) or '') for k in focus_keys):
                    tag_after = (aa.get('tag') or '').upper()
                    role_after = (aa.get('role') or '').lower()
                    if tag_after in ('INPUT', 'TEXTAREA', 'SELECT') or role_after in ('textbox', 'combobox'):
                        changes_detected.append('focus_to_form')
                        if logger:
                            logger.info("Active element changed (to form field)")
                    else:
                        changes_detected.append('focus_changed')
                        if logger:
                            logger.info("Active element changed")
            elif (ab is None) != (aa is None):
                if isinstance(ab, dict) and aa is None:
                    changes_detected.append('focus_blur')
                    if logger:
                        logger.info("Active element changed (blur)")
                elif isinstance(aa, dict) and ab is None:
                    tag_after = (aa.get('tag') or '').upper()
                    role_after = (aa.get('role') or '').lower()
                    if tag_after in ('INPUT', 'TEXTAREA', 'SELECT') or role_after in ('textbox', 'combobox'):
                        changes_detected.append('focus_to_form')
                        if logger:
                            logger.info("Active element changed (to form field)")
                    else:
                        changes_detected.append('focus_changed')
                        if logger:
                            logger.info("Active element changed (None/non-None)")
        except Exception:
            pass

        try:
            if isinstance(ab, dict) and isinstance(aa, dict):
                vb2 = ab.get('value') if isinstance(ab.get('value'), str) else ''
                va2 = aa.get('value') if isinstance(aa.get('value'), str) else ''
                if vb2 != va2:
                    changes_detected.append('active_value_changed')
                    if logger:
                        logger.info("Active element value changed")

                sb2 = ab.get('selectedText') if isinstance(ab.get('selectedText'), str) else ''
                sa2 = aa.get('selectedText') if isinstance(aa.get('selectedText'), str) else ''
                if sb2 != sa2:
                    changes_detected.append('active_selection_changed')
                    if logger:
                        logger.info("Active element selection changed")
        except Exception:
            pass

        hb = state_before.get('hit_target')
        ha = state_after.get('hit_target')
        try:
            if isinstance(hb, dict) and isinstance(ha, dict):
                keys = ('tag', 'id', 'name', 'role', 'ariaLabel', 'href', 'disabled', 'ariaPressed', 'ariaExpanded', 'type', 'value', 'text')
                if any((hb.get(k) or '') != (ha.get(k) or '') for k in keys):
                    changes_detected.append('hit_target_changed')
                    if logger:
                        logger.info("Hit target element changed")
            elif (hb is None) != (ha is None):
                changes_detected.append('hit_target_changed')
                if logger:
                    logger.info("Hit target element changed (None/non-None)")
        except Exception:
            pass
        
        evidence = changes_detected
        if isinstance(action_type, str) and action_type == 'CLICK':
            evidence = [c for c in changes_detected if c != 'focus_blur']
        action_successful = len(evidence) > 0
        
        if isinstance(action_type, str) and action_type in ['CLICK', 'TYPE', 'SELECT', 'KEYBOARD', 'PRESS ENTER'] and not action_successful:
            if state_store is not None:
                try:
                    state_store["_last_changes_detected"] = list(changes_detected)
                    state_store["_last_action_successful"] = False
                except Exception:
                    pass
            if logger:
                logger.warning(f"No clear evidence of success for {action_type} action")
            return False
        if isinstance(action_type, str) and action_type in ['SCROLL UP', 'SCROLL DOWN', 'SCROLL TOP', 'SCROLL BOTTOM']:
            return True
        
        if action_successful:
            if logger:
                logger.info(f"Action '{action_type}' appears SUCCESSFUL - Changes detected: {changes_detected}")
        else:
            if logger:
                logger.warning(f"Action '{action_type}' appears FAILED - No significant page changes detected")

        if state_store is not None:
            try:
                action_results = state_store.setdefault("action_results", [])
                action_results.append({
                    'step': state_store.get("time_step"),
                    'action': action_type,
                    'successful': action_successful,
                    'changes': changes_detected,
                    'timestamp': time.time()
                })
            except Exception:
                pass

        return action_successful
        
    except Exception as e:
        if logger:
            logger.error(f"Error detecting page state change: {e}")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Coordinate request failed: {e}")
        return None


async def find_click_target_by_text(page, text, logger=None):
    try:
        t = text if isinstance(text, str) else ""
        t = t.strip()
        if not t:
            return None
        
        if logger:
            logger.info(f"Attempting to find element by text: '{t}'")

        async def process_locator(locator, source="text"):
            try:
                if await locator.count() > 0:
                    count = await locator.count()
                    for i in range(min(count, 5)):
                        loc = locator.nth(i)
                        if await loc.is_visible():
                            box = await loc.bounding_box()
                            if box and box.get('width', 0) > 0 and box.get('height', 0) > 0:
                                cx = box['x'] + box['width'] / 2
                                cy = box['y'] + box['height'] / 2
                                element = {
                                    'selector': '',
                                    'description': f"Match by {source}: {t}",
                                    'center_point': (cx, cy),
                                    'tag_name': 'unknown',
                                    'box': box
                                }
                                return element, (cx, cy), 'pixel'
            except Exception:
                pass
            return None

        res = await process_locator(page.get_by_text(t, exact=True), "exact_text")
        if res:
            return res

        res = await process_locator(page.get_by_text(t, exact=False), "partial_text")
        if res:
            return res

        if logger:
            logger.info(f"Text match failed, trying fuzzy attribute match for '{t}'")
        
        js_search = """
            (searchText) => {
                if (!searchText) return { found: false };
                const text = searchText.toLowerCase();
                
                function isVisible(el) {
                    const rect = el.getBoundingClientRect();
                    const style = window.getComputedStyle(el);
                    return rect.width > 0 && rect.height > 0 && 
                           style.visibility !== 'hidden' && 
                           style.display !== 'none' && 
                           style.opacity !== '0';
                }

                const attributes = ['aria-label', 'title', 'placeholder', 'alt', 'name', 'value', 'id', 'data-testid', 'aria-description'];
                const selector = attributes.map(attr => `[${attr}]`).join(',');
                const candidates = document.querySelectorAll(selector);
                
                for (const el of candidates) {
                    if (!isVisible(el)) continue;
                    
                    for (const attr of attributes) {
                        const val = el.getAttribute(attr);
                        if (val && val.toLowerCase().includes(text)) {
                            const rect = el.getBoundingClientRect();
                            return {
                                found: true,
                                rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                                matchType: attr,
                                matchValue: val
                            };
                        }
                    }
                }

                const allElements = document.querySelectorAll('body *');
                let bestEl = null;
                let minLen = Infinity;

                for (const el of allElements) {
                    if (el.textContent && el.textContent.toLowerCase().includes(text)) {
                        if (isVisible(el)) {
                            if (el.textContent.length < minLen) {
                                minLen = el.textContent.length;
                                bestEl = el;
                            }
                        }
                    }
                }

                if (bestEl) {
                    const rect = bestEl.getBoundingClientRect();
                    return {
                        found: true,
                        rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                        matchType: 'deep_text',
                        matchValue: bestEl.textContent.trim().substring(0, 50)
                    };
                }

                return { found: false };
            }
        """
        
        result = await page.evaluate(js_search, t)
        
        if result and result.get('found'):
            rect = result['rect']
            cx = rect['x'] + rect['width'] / 2
            cy = rect['y'] + rect['height'] / 2
            element = {
                'selector': '',
                'description': f"Fuzzy match ({result.get('matchType')}): {result.get('matchValue')}",
                'center_point': (cx, cy),
                'tag_name': 'fuzzy_match',
                'box': rect
            }
            if logger:
                logger.info(f"Fuzzy match successful: {element['description']}")
            return element, (cx, cy), 'pixel'

        return None
    except Exception as e:
        if logger:
            logger.warning(f"Find target by text failed: {e}")
        return None


def analyze_previous_action_results(taken_actions, logger=None):
    try:
        if not taken_actions or len(taken_actions) < 2:
            return {"should_terminate": False, "strategy_suggestions": []}
        
        recent_actions = taken_actions[-5:]
        
        failed_actions = [action for action in recent_actions 
                        if isinstance(action, dict) and not action.get('success', True)]
        
        action_types = [action.get('predicted_action', '') for action in recent_actions 
                      if isinstance(action, dict)]
        
        element_descriptions = [action.get('element_description', '') for action in recent_actions 
                              if isinstance(action, dict) and action.get('element_description')]
        
        analysis_result = {
            "should_terminate": False,
            "reason": "",
            "strategy_suggestions": [],
            "alternative_actions": []
        }
        
        if len(failed_actions) >= 3:
            analysis_result["should_terminate"] = True
            analysis_result["reason"] = f"Detected {len(failed_actions)} consecutive failures in recent actions"
            return analysis_result
        
        if len(action_types) >= 3:
            most_common_action = max(set(action_types), key=action_types.count)
            if action_types.count(most_common_action) >= 3:
                repetitive_failures = sum(1 for i, action in enumerate(recent_actions) 
                                        if (isinstance(action, dict) and 
                                            action.get('predicted_action') == most_common_action and 
                                            not action.get('success', True)))
                
                if repetitive_failures >= 2:
                    analysis_result["should_terminate"] = True
                    analysis_result["reason"] = f"Repeated {most_common_action} action failing {repetitive_failures} times"
                    return analysis_result
                else:
                    analysis_result["strategy_suggestions"] = [
                        f"Consider alternatives to repeated {most_common_action} actions",
                        "Try using search functionality if available",
                        "Consider scrolling to find different elements",
                        "Try SELECT action if dropdown elements are available"
                    ]
        
        if len(element_descriptions) >= 3:
            unique_elements = set(element_descriptions)
            if len(unique_elements) == 1:
                last_action = recent_actions[-1]
                if isinstance(last_action, dict) and not last_action.get('success', True):
                    analysis_result["strategy_suggestions"] = [
                        "Current element targeting is not working - try different elements",
                        "Look for alternative navigation paths",
                        "Consider using search or filter functionality",
                        "Try scrolling to reveal more options"
                    ]
        
        last_action = recent_actions[-1] if recent_actions else None
        if isinstance(last_action, dict):
            action_generation = last_action.get('action_generation_response', '').lower()
            
            if ('search' in action_generation and 'not relevant' in action_generation):
                analysis_result["strategy_suggestions"].append(
                    "Reconsider using search functionality - it might be more relevant than initially thought"
                )
            
            if 'select' not in action_generation and 'dropdown' in action_generation:
                analysis_result["alternative_actions"].append("SELECT")
                analysis_result["strategy_suggestions"].append(
                    "Consider using SELECT action for dropdown elements"
                )
        
        if analysis_result["strategy_suggestions"] or analysis_result["alternative_actions"]:
            if logger:
                logger.info("=== REFLECTION ANALYSIS ===")
                logger.info(f"Strategy suggestions: {analysis_result['strategy_suggestions']}")
                logger.info(f"Alternative actions: {analysis_result['alternative_actions']}")
        
        return analysis_result
        
    except Exception as e:
        if logger:
            logger.error(f"Error in reflection analysis: {e}")
        return {"should_terminate": False, "strategy_suggestions": []}


def add_action_to_stack(action_stack, max_stack_size, taken_actions_len, action_type, element_info=None, coordinates=None, logger=None):
    try:
        action_signature = {
            'action': action_type,
            'element': element_info or '',
            'coordinates': coordinates,
            'timestamp': taken_actions_len
        }
        
        action_stack.append(action_signature)
        
        if len(action_stack) > max_stack_size:
            action_stack.pop(0)
            
        if logger:
            logger.debug(f"Added action to stack: {action_signature}")
        
        return action_stack
        
    except Exception as e:
        if logger:
            logger.error(f"Error adding action to stack: {e}")
        return action_stack


def detect_repetitive_actions(action_stack, forbidden_actions, taken_actions, logger=None):
    try:
        if len(action_stack) < 2:
            return {"has_repetition": False, "forbidden_patterns": [], "suggestions": []}
        
        result = {
            "has_repetition": False,
            "forbidden_patterns": [],
            "suggestions": [],
            "repeated_action": None
        }
        
        recent_actions = action_stack[-3:] if len(action_stack) >= 3 else action_stack
        
        if len(recent_actions) >= 2:
            last_action = recent_actions[-1]
            second_last = recent_actions[-2]
            
            if (last_action['action'] == second_last['action'] and 
                last_action['element'] == second_last['element']):
                
                pattern_count = 0
                for i in range(len(recent_actions) - 1):
                    if (recent_actions[i]['action'] == last_action['action'] and
                        recent_actions[i]['element'] == last_action['element']):
                        pattern_count += 1
                
                if pattern_count >= 2:
                    result["has_repetition"] = True
                    result["repeated_action"] = last_action['action']
                    forbidden_pattern = f"{last_action['action']}:{last_action['element']}"
                    result["forbidden_patterns"].append(forbidden_pattern)
                    forbidden_actions.add(forbidden_pattern)
                    
                    result["suggestions"] = [
                        f"You have repeated {last_action['action']} on the same element multiple times.",
                        "This approach is not working. You MUST try a completely different action type.",
                        "Consider: SCROLL, TYPE in search box, SELECT from dropdown, or TERMINATE if stuck."
                    ]
        
        if len(recent_actions) >= 3:
            click_coords = [a['coordinates'] for a in recent_actions if a['action'] == 'CLICK' and a['coordinates']]
            if len(click_coords) >= 2:
                if len(set(click_coords)) == 1:
                    result["has_repetition"] = True
                    result["repeated_action"] = "CLICK"
                    coord_pattern = f"CLICK:{click_coords[0]}"
                    result["forbidden_patterns"].append(coord_pattern)
                    forbidden_actions.add(coord_pattern)
                    
                    result["suggestions"].append(
                        f"You have clicked the same coordinates {click_coords[0]} multiple times. Try a different approach."
                    )
        
        action_types = [a['action'] for a in recent_actions]
        if len(action_types) >= 3:
            most_common = max(set(action_types), key=action_types.count)
            if action_types.count(most_common) >= 3:
                result["has_repetition"] = True
                result["repeated_action"] = most_common
                result["forbidden_patterns"].append(most_common)
                forbidden_actions.add(most_common)
                
                result["suggestions"].append(
                    f"You have used {most_common} action {action_types.count(most_common)} times recently. Try a different action type."
                )

        try:
            if len(taken_actions) >= 3:
                recent_hist = taken_actions[-5:]
                centers = [tuple(a.get('element_center')) for a in recent_hist if a.get('element_center')]
                coords_hist = [tuple(a.get('coordinates')) for a in recent_hist if a.get('coordinates')]
                if centers and len(centers) >= 2:
                    if centers[-1] == centers[-2] and centers.count(centers[-1]) >= 2:
                        result["has_repetition"] = True
                        patt = f"ELEMENT_CENTER:{centers[-1]}"
                        if patt not in result["forbidden_patterns"]:
                            result["forbidden_patterns"].append(patt)
                            forbidden_actions.add(patt)
                        result["suggestions"].append("You are repeating actions on the same element location. Choose a different target.")
                if coords_hist and len(coords_hist) >= 2:
                    if coords_hist[-1] == coords_hist[-2] and coords_hist.count(coords_hist[-1]) >= 2:
                        result["has_repetition"] = True
                        patt = f"COORD:{coords_hist[-1]}"
                        if patt not in result["forbidden_patterns"]:
                            result["forbidden_patterns"].append(patt)
                            forbidden_actions.add(patt)
                        result["suggestions"].append("You are targeting the same coordinates repeatedly. Try a different area.")
        except Exception:
            pass
        
        if result["has_repetition"]:
            if logger:
                logger.warning(f"Repetitive actions detected: {result['forbidden_patterns']}")
            
        return result
        
    except Exception as e:
        if logger:
            logger.error(f"Error detecting repetitive actions: {e}")
        return {"has_repetition": False, "forbidden_patterns": [], "suggestions": []}


def manage_action_history(taken_actions, max_action_history, logger=None):
    try:
        if len(taken_actions) > max_action_history:
            actions_to_remove = len(taken_actions) - max_action_history
            taken_actions = taken_actions[actions_to_remove:]
            if logger:
                logger.info(f"Action history trimmed: removed {actions_to_remove} oldest actions, keeping {len(taken_actions)} recent actions")
        
        if logger:
            logger.debug(f"Action history managed: {len(taken_actions)} actions in history")
        
        return taken_actions
        
    except Exception as e:
        if logger:
            logger.error(f"Error managing action history: {e}")
        return taken_actions


def is_action_forbidden(forbidden_actions, action_type, element_info=None, coordinates=None, logger=None):
    try:
        patterns_to_check = [
            action_type,
            f"{action_type}:{element_info}" if element_info else None,
            f"{action_type}:{coordinates}" if coordinates else None
        ]
        
        for pattern in patterns_to_check:
            if pattern and pattern in forbidden_actions:
                if logger:
                    logger.warning(f"Action forbidden due to repetition: {pattern}")
                return True
                
        return False
        
    except Exception as e:
        if logger:
            logger.error(f"Error checking forbidden action: {e}")
        return False


async def is_page_blocked_or_blank(page, logger=None):
    try:
        current_url = page.url
        
        error_patterns = [
            'chrome-error://',
            'about:blank',
            'about:srcdoc',
            'data:text/html',
            'edge-error://',
            'about:neterror'
        ]
        
        for pattern in error_patterns:
            if current_url.startswith(pattern):
                if logger:
                    logger.warning(f"Detected error URL: {current_url}")
                return True
        
        error_keywords = ['ERR_', 'error', 'blocked', 'denied']
        if any(keyword in current_url for keyword in error_keywords):
            if logger:
                logger.warning(f"Detected error keyword in URL: {current_url}")
            return True
        
        page_title = await page.title()
        page_content = await page.content()
        
        if len(page_content) < 100:
            if logger:
                logger.warning(f"Page content too short ({len(page_content)} chars)")
            return True
        
        error_titles = [
            'Access Denied',
            'Forbidden',
            '403',
            '404',
            '500',
            'Error',
            'Not Found',
            'Unavailable',
            'Failed to load'
        ]
        
        if any(error in page_title for error in error_titles):
            if logger:
                logger.warning(f"Detected error in page title: {page_title}")
            return True
        
        body_text = await page.evaluate('() => document.body?.innerText || ""')
        error_messages = [
            'Sorry, you have been blocked',
            'you have been blocked',
            'ERR_HTTP2_PROTOCOL_ERROR',
            'ERR_CONNECTION',
            'Access Denied',
            'Forbidden',
            'This site can\'t be reached',
            'Unable to connect',
            'The page isn\'t working',
            'Cloudflare Ray ID'
        ]
        
        for error_msg in error_messages:
            if error_msg.lower() in body_text.lower():
                if logger:
                    logger.warning(f"Detected error message in page: {error_msg}")
                return True
        
        return False
        
    except Exception as e:
        if logger:
            logger.error(f"Error checking page status: {e}")
        return True


