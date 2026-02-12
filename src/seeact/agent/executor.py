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
import asyncio
import random
import traceback
import re
from seeact.browser.dom import (
    dom_select_option, get_all_select_elements, decide_form_element,
    extract_typeable_elements, verify_last_typing, clear_active_field,
    looks_like_api_endpoint, choose_field_with_llm, dom_select_by_selector,
    extract_selectable_elements
)
from seeact.browser.recovery import find_click_target_by_text

async def perform_action(
    agent,
    target_element=None,
    action_name=None,
    value=None,
    target_coordinates=None,
    element_repr=None,
    field_name=None,
    action_description=None,
    clear_first: bool = True,
    press_enter_after: bool = False
):
    element_info = element_repr or (target_element.get('description') if target_element else None)
    coordinates = None
    if target_coordinates:
        if isinstance(target_coordinates, dict):
            coordinates = (round(target_coordinates["x"]), round(target_coordinates["y"]))
        elif isinstance(target_coordinates, (tuple, list)) and len(target_coordinates) >= 2:
            coordinates = (round(target_coordinates[0]), round(target_coordinates[1]))
        else:
            agent.logger.error(f"Invalid target_coordinates format: {type(target_coordinates)} - {target_coordinates}")
            coordinates = None
    elif target_element and target_element.get('center_point'):
        coords = target_element.get('center_point')
        coordinates = (round(coords[0]), round(coords[1]))
    
    agent._add_action_to_stack(action_name, element_info, coordinates)
    
    if target_element is not None and isinstance(target_element, dict):
        selector = target_element.get('selector')
        element_repr = target_element.get('description', 'Unknown element')
    else:
        selector = None

    page = agent.page

    if action_name == "CLICK":
        click_failed = False
        failure_reason = ""
        
        if target_coordinates:
            try:
                if isinstance(target_coordinates, dict):
                    llm_x, llm_y = target_coordinates["x"], target_coordinates["y"]
                elif isinstance(target_coordinates, (tuple, list)) and len(target_coordinates) >= 2:
                    llm_x, llm_y = target_coordinates[0], target_coordinates[1]
                else:
                    agent.logger.error(f"Invalid target_coordinates format: {type(target_coordinates)} - {target_coordinates}")
                    click_failed = True
                    failure_reason = f"Invalid coordinate format: {type(target_coordinates)}"
                    raise ValueError(failure_reason)
                
                if not isinstance(llm_x, (int, float)) or not isinstance(llm_y, (int, float)):
                    raise ValueError(f"Coordinates must be numeric: x={type(llm_x)}, y={type(llm_y)}")
                
                agent.logger.debug(f"🔍 [CLICK] Raw coordinates: x={llm_x} (type: {type(llm_x)}), y={llm_y} (type: {type(llm_y)})")
                
                agent.last_click_coordinates = (round(llm_x), round(llm_y))
                
                try:
                    if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                        scaled_x, scaled_y = agent.map_normalized_to_pixels(llm_x, llm_y)
                    else:
                        scaled_x, scaled_y = agent.normalize_coords(llm_x, llm_y, None)
                    agent.logger.info(f"🎯 [CLICK] Coordinate Mapping: input({llm_x},{llm_y}) → Viewport({scaled_x},{scaled_y})")
                except Exception as norm_error:
                    raise ValueError(f"Coordinate normalization failed: {norm_error}")
                
                viewport_width = agent.config.get('browser', {}).get('viewport', {}).get('width', 1500)
                viewport_height = agent.config.get('browser', {}).get('viewport', {}).get('height', 1200)
                if not (0 <= scaled_x < viewport_width and 0 <= scaled_y < viewport_height):
                    agent.logger.warning(f"⚠️ [CLICK] Normalized coordinates ({scaled_x}, {scaled_y}) outside viewport ({viewport_width}x{viewport_height}), clamping...")
                    scaled_x = max(0, min(viewport_width - 1, scaled_x))
                    scaled_y = max(0, min(viewport_height - 1, scaled_y))
                
                delay = random.randint(50, 150)
                final_x, final_y = round(scaled_x), round(scaled_y)
                agent.logger.debug(f"🖱️ [CLICK] Executing click at ({final_x}, {final_y}) with delay={delay}ms")
                
                try:
                    await agent.page.mouse.click(final_x, final_y, delay=delay)
                    agent.last_click_viewport_coords = (final_x, final_y)
                    agent.logger.debug(f"✅ [CLICK] Playwright click executed successfully")
                except Exception as click_error:
                    agent.logger.error(f"❌ [CLICK] Playwright click failed: {type(click_error).__name__}: {click_error} | Trying mouse down/up fallback")
                    agent.logger.debug(f"   Click params: x={final_x}, y={final_y}, delay={delay}")
                    try:
                        await agent.page.mouse.move(final_x, final_y)
                        await agent.page.mouse.down()
                        await asyncio.sleep(0.05)
                        await agent.page.mouse.up()
                        agent.last_click_viewport_coords = (final_x, final_y)
                        agent.logger.debug(f"✅ [CLICK] Fallback mouse down/up executed successfully")
                    except Exception as click_error2:
                        agent.logger.error(f"❌ [CLICK] Fallback mouse down/up failed: {type(click_error2).__name__}: {click_error2}")
                        raise
                
                await asyncio.sleep(0.8)
                
                agent.logger.info(f"✅ [CLICK] Successfully clicked at viewport coordinates ({final_x}, {final_y})")
                return f"Clicked at coordinates ({final_x}, {final_y})"
                
            except Exception as e:
                click_failed = True
                failure_reason = f"Coordinate click failed - {type(e).__name__}: {str(e)}"
                agent.logger.error(f"❌ [CLICK] {failure_reason}")
                agent.logger.debug(traceback.format_exc())
        else:
            click_failed = True
            failure_reason = "No coordinates provided for CLICK action"
            agent.logger.error(failure_reason)
        
        if click_failed and element_repr:
            agent.logger.info(f"⚠️ [CLICK] Coordinate click failed/missing, attempting fallback to text match for: {element_repr}")
            found_elem, _, _ = await find_click_target_by_text(agent.page, element_repr, logger=agent.logger)
            if found_elem and found_elem.get('box'):
                 box = found_elem['box']
                 cx = box['x'] + box['width'] / 2
                 cy = box['y'] + box['height'] / 2
                 agent.logger.info(f"✅ [CLICK] Fallback found element by text at ({cx}, {cy})")
                 await agent.page.mouse.click(cx, cy)
                 return f"Clicked element found by text: {element_repr}"
        
        if click_failed:
            return f"FAILED: {failure_reason}"

    elif action_name == "HOVER":
        hover_failed = False
        failure_reason = ""
        
        if target_coordinates and not target_element:
            try:
                delay = random.randint(50, 150)
                if isinstance(target_coordinates, dict):
                    nx, ny = target_coordinates["x"], target_coordinates["y"]
                elif isinstance(target_coordinates, (tuple, list)) and len(target_coordinates) >= 2:
                    nx, ny = target_coordinates[0], target_coordinates[1]
                else:
                    agent.logger.error(f"Invalid target_coordinates format for HOVER action: {type(target_coordinates)} - {target_coordinates}")
                    hover_failed = True
                    failure_reason = f"Invalid coordinate format: {type(target_coordinates)}"
                    raise ValueError(failure_reason)
                try:
                    if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                        nx, ny = agent.map_normalized_to_pixels(nx, ny)
                    else:
                        nx, ny = agent.normalize_coords(nx, ny, None)
                except Exception:
                    await agent.take_screenshot()
                    try:
                        if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                            nx, ny = agent.map_normalized_to_pixels(nx, ny)
                        else:
                            nx, ny = agent.normalize_coords(nx, ny, None)
                    except Exception:
                        raise ValueError("Coordinate normalization failed for HOVER")
                await agent.page.mouse.hover(round(nx), round(ny), delay=delay)
                agent.logger.info(f"Hovered at coordinates ({nx}, {ny})")
                return f"SUCCESS: Hovered at coordinates ({nx}, {ny})"
            except Exception as e:
                hover_failed = True
                failure_reason = f"Coordinate hover failed - {e}"
        elif target_element:
            try:
                if selector:
                    await selector.hover(timeout=10000)
                    agent.logger.info(f"Hovered over element: {element_repr}")
                    return f"SUCCESS: Hovered over element: {element_repr}"
                elif isinstance(target_element, dict) and target_element.get('center_point'):
                    coords = target_element.get('center_point')
                    delay = random.randint(50, 150)
                    await agent.page.mouse.hover(round(coords[0]), round(coords[1]), delay=delay)
                    agent.logger.info(f"Hovered over element at ({coords[0]}, {coords[1]}): {element_repr}")
                    return f"SUCCESS: Hovered over element at ({coords[0]}, {coords[1]}): {element_repr}"
                else:
                    raise ValueError("No selector or coordinates available for target element in HOVER")
            except Exception as e:
                hover_failed = True
                failure_reason = f"Hover action failed - {e}"
                agent.logger.warning(failure_reason)
        elif selector:
            try:
                await selector.hover(timeout=10000)
                agent.logger.info(f"Hovered over element: {element_repr}")
                return f"SUCCESS: Hovered over element: {element_repr}"
            except Exception as e:
                hover_failed = True
                failure_reason = f"Hover action failed - {e}"
                agent.logger.warning(failure_reason)
        else:
            hover_failed = True
            failure_reason = "No valid target for HOVER action"
            agent.logger.error(failure_reason)
        
        if hover_failed:
            return f"FAILED: {failure_reason}"

    elif action_name == "TYPE":
        if not field_name:
            field_name = 'unknown'
        agent.logger.info(f"🎯 [TYPE] Starting | Field: '{field_name}' | Value: '{value}' | Coords: {target_coordinates} | EnterAfter: {press_enter_after}")
        coords = None
        if isinstance(target_coordinates, dict):
            coords = (target_coordinates["x"], target_coordinates["y"])
        elif isinstance(target_coordinates, (tuple, list)) and len(target_coordinates) >= 2:
            coords = (target_coordinates[0], target_coordinates[1])
        
        attempts = 0
        last_error = None
        
        if coords:
            try:
                nx, ny = coords
                try:
                    if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                        nx, ny = agent.map_normalized_to_pixels(nx, ny)
                    else:
                        nx, ny = agent.normalize_coords(nx, ny, None)
                except Exception:
                    if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                        nx, ny = agent.map_normalized_to_pixels(coords[0], coords[1])
                    else:
                        nx, ny = agent.normalize_coords(coords[0], coords[1], None)
                
                delay = random.randint(50, 150)
                agent.logger.info(f"🖱️ [TYPE] Clicking coordinates ({nx}, {ny}) to focus")
                await agent.page.mouse.click(round(nx), round(ny), delay=delay)
                
                if clear_first:
                    await clear_active_field(agent.page)
                
                await agent.page.keyboard.type(value)
                
                if await verify_last_typing(agent.page, value):
                    agent.logger.info(f"✅ [TYPE L1/3] Success | Method: keyboard | Coords: ({nx}, {ny})")
                    if press_enter_after:
                        await agent.page.keyboard.press('Enter')
                        return f"SUCCESS: Typed '{value}' via keyboard and pressed Enter"
                    return f"SUCCESS: Typed '{value}' via keyboard"
                else:
                    attempts += 1
                    last_error = "Typing verification failed"
                    agent.logger.warning(f"⚠️ [TYPE L1/3] Failed | Method: keyboard | Error: {last_error} | Trying element selector")
            except Exception as e:
                attempts += 1
                last_error = str(e)
                agent.logger.warning(f"⚠️ [TYPE L1/3] Failed | Method: keyboard | Error: {str(e)[:120]} | Trying element selector")
        else:
            agent.logger.info(f"⚠️ [TYPE L1/3] Skipped | Reason: No coordinates provided | Trying element selector")
            attempts += 1

        if selector:
            try:
                await selector.click(timeout=10000)
                try:
                    await selector.fill(value, timeout=10000)
                except Exception:
                    if clear_first:
                        await clear_active_field(agent.page)
                    await agent.page.keyboard.type(value)
                if await verify_last_typing(agent.page, value):
                    agent.logger.info(f"✅ [TYPE ELEM] Success | Method: selector+keyboard | Element: {element_repr}")
                    if press_enter_after:
                        try:
                            await selector.press('Enter', timeout=5000)
                        except Exception:
                            await agent.page.keyboard.press('Enter')
                        return f"SUCCESS: Typed '{value}' in {element_repr} and pressed Enter"
                    return f"SUCCESS: Typed '{value}' in {element_repr}"
                else:
                    attempts += 1
                    last_error = "Typing verification failed (selector path)"
                    agent.logger.warning(f"⚠️ [TYPE ELEM] Failed | Reason: {last_error} | Trying nearest input")
            except Exception as e:
                attempts += 1
                last_error = str(e)
                agent.logger.warning(f"⚠️ [TYPE ELEM] Exception | Error: {str(e)[:160]} | Trying nearest input")

        try:
            if coords:
                all_inputs = await extract_typeable_elements(agent.page, logger=agent.logger)
                if all_inputs:
                    def dist(a, b):
                        return ((a['center']['x'] - b[0]) ** 2 + (a['center']['y'] - b[1]) ** 2) ** 0.5
                    sorted_inputs = sorted(all_inputs, key=lambda i: dist(i, coords))
                    top_candidates = sorted_inputs[:3]
                    chosen_input = top_candidates[0]
                    if len(top_candidates) > 1:
                        chosen_input = await choose_field_with_llm(
                            agent.engine,
                            agent.model,
                            agent.logger,
                            top_candidates,
                            "TYPE",
                            value,
                            f"Find '{field_name}' field for: {agent.tasks[-1] if agent.tasks else 'Unknown task'}"
                        ) or top_candidates[0]
                    center = chosen_input['center']
                    delay = random.randint(50, 150)
                    sel_str = chosen_input.get('selector')
                    used_locator = False
                    if isinstance(sel_str, str) and sel_str.strip():
                        try:
                            await agent.page.locator(sel_str).click(timeout=10000)
                            used_locator = True
                        except Exception:
                            used_locator = False
                    if not used_locator:
                        await agent.page.mouse.click(round(center['x']), round(center['y']), delay=delay)
                    if clear_first:
                        await clear_active_field(agent.page)
                    await agent.page.keyboard.type(value)
                    agent.logger.info(f"✅ [TYPE L2/3] Success | Method: nearest+keyboard | Field: {chosen_input.get('label') or chosen_input.get('placeholder') or 'Unnamed'}")
                    if press_enter_after:
                        await agent.page.keyboard.press('Enter')
                        return f"SUCCESS: Typed '{value}' via keyboard and pressed Enter"
                    return f"SUCCESS: Typed '{value}' via keyboard"
                else:
                    last_error = "No input fields found"
                    agent.logger.warning(f"⚠️ [TYPE L2/3] Failed | Reason: {last_error} | Trying GUI grounding")
                    attempts += 1
            else:
                all_inputs = await extract_typeable_elements(agent.page, logger=agent.logger)
                if all_inputs:
                    def score_input(el):
                        label = (el.get('label') or el.get('placeholder') or el.get('name') or '').lower()
                        fn = (field_name or '').lower()
                        s = 0
                        if fn and label:
                            if fn in label:
                                s += 3
                        try:
                            w = float(el.get('width', el.get('rect', {}).get('width', 0)) or 0)
                            h = float(el.get('height', el.get('rect', {}).get('height', 0)) or 0)
                            if w*h > 0:
                                s += 1
                        except Exception:
                            pass
                        return -s
                    sorted_inputs = sorted(all_inputs, key=score_input)
                    top_candidates = sorted_inputs[:3] if len(sorted_inputs) > 0 else []
                    if not top_candidates:
                        attempts += 1
                    else:
                        chosen_input = top_candidates[0]
                        if len(top_candidates) > 1:
                            chosen_input = await choose_field_with_llm(
                                agent.engine,
                                agent.model,
                                agent.logger,
                                top_candidates,
                                "TYPE",
                                value,
                                f"Find '{field_name}' field for: {agent.tasks[-1] if agent.tasks else 'Unknown task'}"
                            ) or top_candidates[0]
                        center = chosen_input.get('center') or {
                            'x': chosen_input.get('center_point', (agent.config.get('browser', {}).get('viewport', {}).get('width', 1280)//2, agent.config.get('browser', {}).get('viewport', {}).get('height', 720)//2))[0],
                            'y': chosen_input.get('center_point', (agent.config.get('browser', {}).get('viewport', {}).get('width', 1280)//2, agent.config.get('browser', {}).get('viewport', {}).get('height', 720)//2))[1]
                        }
                        delay = random.randint(50, 150)
                        sel_str = chosen_input.get('selector')
                        used_locator = False
                        if isinstance(sel_str, str) and sel_str.strip():
                            try:
                                await agent.page.locator(sel_str).click(timeout=10000)
                                used_locator = True
                            except Exception:
                                used_locator = False
                        if not used_locator:
                            await agent.page.mouse.click(round(center['x']), round(center['y']), delay=delay)
                        if clear_first:
                            await clear_active_field(agent.page)
                        await agent.page.keyboard.type(value)
                        agent.logger.info(f"✅ [TYPE L2b/3] Success | Method: semantic+keyboard | Field: {chosen_input.get('label') or chosen_input.get('placeholder') or 'Unnamed'}")
                        return f"SUCCESS: Typed '{value}' via keyboard"
                else:
                    attempts += 1
        except Exception as e:
            attempts += 1
            last_error = str(e)
            agent.logger.warning(f"⚠️ [TYPE L2/3] Exception | Method: nearest+keyboard | Error: {str(e)[:120]}")
        return f"FAILED: TYPE after {attempts} attempts - {last_error}"

    elif action_name == "KEYBOARD":
        if not value:
            return "FAILED: KEYBOARD requires text or CLEAR"
        content = value.strip()
        try:
            coords = None
            if isinstance(target_coordinates, dict):
                coords = (target_coordinates["x"], target_coordinates["y"]) 
            elif isinstance(target_coordinates, (tuple, list)) and len(target_coordinates) >= 2:
                coords = (target_coordinates[0], target_coordinates[1])
            if coords:
                delay = random.randint(50, 150)
                try:
                    if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                        nx, ny = agent.map_normalized_to_pixels(coords[0], coords[1])
                    else:
                        nx, ny = agent.normalize_coords(coords[0], coords[1], None)
                except Exception:
                    await agent.take_screenshot()
                    if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                        nx, ny = agent.map_normalized_to_pixels(coords[0], coords[1])
                    else:
                        nx, ny = agent.normalize_coords(coords[0], coords[1], None)
                await agent.page.mouse.click(round(nx), round(ny), delay=delay)
            if content.upper() == "CLEAR":
                await clear_active_field(agent.page)
                return "SUCCESS: Cleared field via keyboard"
            if clear_first:
                await clear_active_field(agent.page)
            known_codes = {
                "Enter", "Tab", "Escape", "Backspace", "Delete", "Home", "End",
                "PageUp", "PageDown", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"
            }
            is_code_like = bool(re.match(r"^(Key[A-Z]|Digit[0-9]|F[0-9]{1,2}|Arrow(Up|Down|Left|Right)|Page(Up|Down)|Enter|Tab|Escape|Backspace|Delete|Home|End)$", content))
            if content in known_codes or is_code_like:
                await agent.page.keyboard.press(content)
                return f"SUCCESS: Pressed '{content}' via keyboard"
            await agent.page.keyboard.type(content)
            return f"SUCCESS: Typed '{content}' via keyboard"
        except Exception as e:
            agent.logger.debug(traceback.format_exc())
            return f"FAILED: KEYBOARD action failed - {e}"

    elif action_name == "SCROLL UP":
        try:
            vp = getattr(page, 'viewport_size', None) or agent.config.get('browser', {}).get('viewport', {})
            vh = vp.get('height', 720) if isinstance(vp, dict) else (vp.height if hasattr(vp, 'height') else 720)
            delta = -max(100, int(vh * 0.6))
            if isinstance(target_coordinates, dict) or (isinstance(target_coordinates, (tuple, list)) and len(target_coordinates) >= 2):
                if isinstance(target_coordinates, dict):
                    nx, ny = target_coordinates.get('x'), target_coordinates.get('y')
                else:
                    nx, ny = target_coordinates[0], target_coordinates[1]
                try:
                    if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                        nx, ny = agent.map_normalized_to_pixels(nx, ny)
                    else:
                        nx, ny = agent.normalize_coords(nx, ny, None)
                except Exception:
                    await agent.take_screenshot()
                    try:
                        if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                            nx, ny = agent.map_normalized_to_pixels(nx, ny)
                        else:
                            nx, ny = agent.normalize_coords(nx, ny, None)
                    except Exception:
                        return "FAILED: Coordinate normalization failed"
                await page.evaluate(
                    "(params) => {\n"+
                    "  const { x, y, d } = params;\n"+
                    "  const el = document.elementFromPoint(x, y);\n"+
                    "  function scrollableAncestor(node){\n"+
                    "    let n = node;\n"+
                    "    while(n && n !== document.body && n !== document.documentElement){\n"+
                    "      const s = getComputedStyle(n);\n"+
                    "      const oy = s.overflowY;\n"+
                    "      if((oy === 'auto' || oy === 'scroll' || oy === 'overlay') && n.scrollHeight > n.clientHeight){\n"+
                    "        return n;\n"+
                    "      }\n"+
                    "      n = n.parentElement;\n"+
                    "    }\n"+
                    "    return window;\n"+
                    "  }\n"+
                    "  function scrollWin(d){\n"+
                    "    try{ window.scrollBy(0,d); }catch(e){}\n"+
                    "    try{ document.documentElement.scrollBy(0,d); }catch(e){}\n"+
                    "    try{ document.documentElement.scrollTop += d; }catch(e){}\n"+
                    "    try{ document.body.scrollTop += d; }catch(e){}\n"+
                    "  }\n"+
                    "  const sc = scrollableAncestor(el);\n"+
                    "  if(sc === window){ scrollWin(d); } else { try{ sc.scrollBy(0,d);}catch(e){} try{ sc.scrollTop += d;}catch(e){} }\n"+
                    "}",
                    {"x": nx, "y": ny, "d": delta}
                )
            else:
                await page.evaluate(
                    "(d) => {\n"+
                    "  try{ window.scrollBy(0,d);}catch(e){}\n"+
                    "  try{ document.documentElement.scrollBy(0,d);}catch(e){}\n"+
                    "  try{ document.documentElement.scrollTop += d;}catch(e){}\n"+
                    "  try{ document.body.scrollTop += d;}catch(e){}\n"+
                    "}",
                    delta
                )
            agent.logger.info("Scrolled up")
            return "Scrolled up"
        except Exception as e:
            agent.logger.error(f"Failed to scroll up: {e}")
            return f"FAILED: Scroll up - {e}"
    elif action_name == "SCROLL DOWN":
        try:
            vp = getattr(page, 'viewport_size', None) or agent.config.get('browser', {}).get('viewport', {})
            vh = vp.get('height', 720) if isinstance(vp, dict) else (vp.height if hasattr(vp, 'height') else 720)
            delta = max(100, int(vh * 0.6))
            if isinstance(target_coordinates, dict) or (isinstance(target_coordinates, (tuple, list)) and len(target_coordinates) >= 2):
                if isinstance(target_coordinates, dict):
                    nx, ny = target_coordinates.get('x'), target_coordinates.get('y')
                else:
                    nx, ny = target_coordinates[0], target_coordinates[1]
                try:
                    if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                        nx, ny = agent.map_normalized_to_pixels(nx, ny)
                    else:
                        nx, ny = agent.normalize_coords(nx, ny, None)
                except Exception:
                    await agent.take_screenshot()
                    try:
                        if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                            nx, ny = agent.map_normalized_to_pixels(nx, ny)
                        else:
                            nx, ny = agent.normalize_coords(nx, ny, None)
                    except Exception:
                        return "FAILED: Coordinate normalization failed"
                await page.evaluate(
                    "(params) => {\n"+
                    "  const { x, y, d } = params;\n"+
                    "  const el = document.elementFromPoint(x, y);\n"+
                    "  function scrollableAncestor(node){\n"+
                    "    let n = node;\n"+
                    "    while(n && n !== document.body && n !== document.documentElement){\n"+
                    "      const s = getComputedStyle(n);\n"+
                    "      const oy = s.overflowY;\n"+
                    "      if((oy === 'auto' || oy === 'scroll' || oy === 'overlay') && n.scrollHeight > n.clientHeight){\n"+
                    "        return n;\n"+
                    "      }\n"+
                    "      n = n.parentElement;\n"+
                    "    }\n"+
                    "    return window;\n"+
                    "  }\n"+
                    "  function scrollWin(d){\n"+
                    "    try{ window.scrollBy(0,d); }catch(e){}\n"+
                    "    try{ document.documentElement.scrollBy(0,d); }catch(e){}\n"+
                    "    try{ document.documentElement.scrollTop += d; }catch(e){}\n"+
                    "    try{ document.body.scrollTop += d; }catch(e){}\n"+
                    "  }\n"+
                    "  const sc = scrollableAncestor(el);\n"+
                    "  if(sc === window){ scrollWin(d); } else { try{ sc.scrollBy(0,d);}catch(e){} try{ sc.scrollTop += d;}catch(e){} }\n"+
                    "}",
                    {"x": nx, "y": ny, "d": delta}
                )
            else:
                await page.evaluate(
                    "(d) => {\n"+
                    "  try{ window.scrollBy(0,d);}catch(e){}\n"+
                    "  try{ document.documentElement.scrollBy(0,d);}catch(e){}\n"+
                    "  try{ document.documentElement.scrollTop += d;}catch(e){}\n"+
                    "  try{ document.body.scrollTop += d;}catch(e){}\n"+
                    "}",
                    delta
                )
            agent.logger.info("Scrolled down")
            return "Scrolled down"
        except Exception as e:
            agent.logger.error(f"Failed to scroll down: {e}")
            return f"FAILED: Scroll down - {e}"
    elif action_name == "SCROLL TOP":
        try:
            if isinstance(target_coordinates, dict) or (isinstance(target_coordinates, (tuple, list)) and len(target_coordinates) >= 2):
                if isinstance(target_coordinates, dict):
                    nx, ny = target_coordinates.get('x'), target_coordinates.get('y')
                else:
                    nx, ny = target_coordinates[0], target_coordinates[1]
                try:
                    if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                        nx, ny = agent.map_normalized_to_pixels(nx, ny)
                    else:
                        nx, ny = agent.normalize_coords(nx, ny, None)
                except Exception:
                    await agent.take_screenshot()
                    try:
                        if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                            nx, ny = agent.map_normalized_to_pixels(nx, ny)
                        else:
                            nx, ny = agent.normalize_coords(nx, ny, None)
                    except Exception:
                        return "FAILED: Coordinate normalization failed"
                await page.evaluate(
                    "(params) => {\n"+
                    "  const { x, y } = params;\n"+
                    "  const el = document.elementFromPoint(x, y);\n"+
                    "  function scrollableAncestor(node){\n"+
                    "    let n = node;\n"+
                    "    while(n && n !== document.body && n !== document.documentElement){\n"+
                    "      const s = getComputedStyle(n);\n"+
                    "      const oy = s.overflowY;\n"+
                    "      if((oy === 'auto' || oy === 'scroll' || oy === 'overlay') && n.scrollHeight > n.clientHeight){\n"+
                    "        return n;\n"+
                    "      }\n"+
                    "      n = n.parentElement;\n"+
                    "    }\n"+
                    "    return window;\n"+
                    "  }\n"+
                    "  function scrollTopWin(){\n"+
                    "    try{ window.scrollTo(0,0);}catch(e){}\n"+
                    "    try{ document.documentElement.scrollTo(0,0);}catch(e){}\n"+
                    "    try{ document.documentElement.scrollTop = 0;}catch(e){}\n"+
                    "    try{ document.body.scrollTop = 0;}catch(e){}\n"+
                    "  }\n"+
                    "  const sc = scrollableAncestor(el);\n"+
                    "  if(sc === window){ scrollTopWin(); } else { try{ sc.scrollTop = 0;}catch(e){} }\n"+
                    "}",
                    {"x": nx, "y": ny}
                )
            else:
                await page.evaluate(
                    "() => {\n"+
                    "  try{ window.scrollTo(0,0);}catch(e){}\n"+
                    "  try{ document.documentElement.scrollTo(0,0);}catch(e){}\n"+
                    "  try{ document.documentElement.scrollTop = 0;}catch(e){}\n"+
                    "  try{ document.body.scrollTop = 0;}catch(e){}\n"+
                    "}"
                )
            agent.logger.info("Scrolled to top")
            return "Scrolled to top"
        except Exception as e:
            agent.logger.error(f"Failed to scroll to top: {e}")
            return f"FAILED: Scroll to top - {e}"
    elif action_name == "SCROLL BOTTOM":
        try:
            if isinstance(target_coordinates, dict) or (isinstance(target_coordinates, (tuple, list)) and len(target_coordinates) >= 2):
                if isinstance(target_coordinates, dict):
                    nx, ny = target_coordinates.get('x'), target_coordinates.get('y')
                else:
                    nx, ny = target_coordinates[0], target_coordinates[1]
                try:
                    if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                        nx, ny = agent.map_normalized_to_pixels(nx, ny)
                    else:
                        nx, ny = agent.normalize_coords(nx, ny, None)
                except Exception:
                    await agent.take_screenshot()
                    try:
                        if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                            nx, ny = agent.map_normalized_to_pixels(nx, ny)
                        else:
                            nx, ny = agent.normalize_coords(nx, ny, None)
                    except Exception:
                        return "FAILED: Coordinate normalization failed"
                await page.evaluate(
                    "(params) => {\n"+
                    "  const { x, y } = params;\n"+
                    "  const el = document.elementFromPoint(x, y);\n"+
                    "  function scrollableAncestor(node){\n"+
                    "    let n = node;\n"+
                    "    while(n && n !== document.body && n !== document.documentElement){\n"+
                    "      const s = getComputedStyle(n);\n"+
                    "      const oy = s.overflowY;\n"+
                    "      if((oy === 'auto' || oy === 'scroll' || oy === 'overlay') && n.scrollHeight > n.clientHeight){\n"+
                    "        return n;\n"+
                    "      }\n"+
                    "      n = n.parentElement;\n"+
                    "    }\n"+
                    "    return window;\n"+
                    "  }\n"+
                    "  function scrollBottomWin(){\n"+
                    "    const h = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight);\n"+
                    "    try{ window.scrollTo(0, h);}catch(e){}\n"+
                    "    try{ document.documentElement.scrollTo(0, h);}catch(e){}\n"+
                    "    try{ document.documentElement.scrollTop = h;}catch(e){}\n"+
                    "    try{ document.body.scrollTop = h;}catch(e){}\n"+
                    "  }\n"+
                    "  const sc = scrollableAncestor(el);\n"+
                    "  if(sc === window){ scrollBottomWin(); } else { try{ sc.scrollTop = sc.scrollHeight;}catch(e){} }\n"+
                    "}",
                    {"x": nx, "y": ny}
                )
            else:
                await page.evaluate(
                    "() => {\n"+
                    "  const h = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight);\n"+
                    "  try{ window.scrollTo(0, h);}catch(e){}\n"+
                    "  try{ document.documentElement.scrollTo(0, h);}catch(e){}\n"+
                    "  try{ document.documentElement.scrollTop = h;}catch(e){}\n"+
                    "  try{ document.body.scrollTop = h;}catch(e){}\n"+
                    "}"
                )
            agent.logger.info("Scrolled to bottom")
            return "Scrolled to bottom"
        except Exception as e:
            agent.logger.error(f"Failed to scroll to bottom: {e}")
            return f"FAILED: Scroll to bottom - {e}"
    elif action_name == "PRESS HOME":
        await page.keyboard.press('Home')
        agent.logger.info("Pressed Home key")
        return "Pressed Home key"
    elif action_name == "PRESS END":
        await page.keyboard.press('End')
        agent.logger.info("Pressed End key")
        return "Pressed End key"
    elif action_name == "PRESS PAGEUP":
        await page.keyboard.press('PageUp')
        agent.logger.info("Pressed PageUp key")
        return "Pressed PageUp key"
    elif action_name == "PRESS PAGEDOWN":
        await page.keyboard.press('PageDown')
        agent.logger.info("Pressed PageDown key")
        return "Pressed PageDown key"
    elif action_name == "NEW TAB":
        if (agent.session_control and isinstance(agent.session_control, dict) and 
            agent.session_control.get('context') and 
            hasattr(agent.session_control['context'], 'pages')):
            current_pages = len(agent.session_control['context'].pages)
            max_pages = 3
            
            if current_pages >= max_pages:
                agent.logger.warning(f"Page limit reached ({current_pages}/{max_pages}). Cannot create new tab.")
                return f"SKIPPED: Page limit reached ({current_pages}/{max_pages})"
            
            new_page = await agent.session_control['context'].new_page()
            agent.logger.info(f"Opened a new tab (total pages: {len(agent.session_control['context'].pages)})")
            return f"Opened a new tab (total pages: {len(agent.session_control['context'].pages)})"
        else:
            agent.logger.warning("Cannot create new tab: session_control or context not properly initialized")
            return "FAILED: Cannot create new tab: session_control or context not properly initialized"
    elif action_name == "CLOSE TAB":
        await page.close()
        agent.logger.info("Closed the current tab")
        return "Closed the current tab"
    elif action_name == "GO BACK":
        await page.go_back()
        agent.logger.info("Navigated back")
        return "Navigated back"
    elif action_name == "GO FORWARD":
        await page.go_forward()
        agent.logger.info("Navigated forward")
        return "Navigated forward"
    elif action_name == "GOTO" and value:
        try:
            if looks_like_api_endpoint(value):
                agent.logger.error(f"Blocked GOTO to non-HTML endpoint: {value}")
                return f"FAILED: GOTO blocked for API/GraphQL endpoint"
            resp = await page.goto(value, wait_until="load")
            agent.logger.info(f"Navigated to {value}")
            ct = ""
            try:
                if resp:
                    h = resp.headers
                    ct = h.get("content-type", "").lower()
            except Exception:
                ct = ""
            if ct and "text/html" not in ct:
                agent.logger.error(f"Non-HTML content-type '{ct}' after GOTO {value}; reverting")
                try:
                    await page.go_back()
                except Exception:
                    pass
                return f"FAILED: Navigated to non-HTML endpoint ({ct})"
            if await agent._is_page_blocked_or_blank():
                agent.logger.error(f"⛔ Page is blocked or blank after navigating to {value}")
                agent.complete_flag = True
                return f"FAILED: Page is blocked or blank at {value}"
            return f"Navigated to {value}"
        except Exception as e:
            agent.logger.error(f"GOTO action failed: {e}")
            return f"FAILED: Navigation to {value} failed - {e}"
    elif action_name == "PRESS ENTER" and selector:
        if selector == "pixel_coordinates":
            await agent.page.keyboard.press('Enter')
            agent.logger.info("Pressed Enter without using coordinates (keyboard-only)")
            return "SUCCESS: Pressed Enter (keyboard-only)"
        await selector.press('Enter')
        agent.logger.info(f"Pressed Enter on element: {element_repr}")
        return f"Pressed Enter on element: {element_repr}"
    elif action_name == "WAIT":
        try:
            duration = 1
            if isinstance(value, (int, float)):
                duration = max(0, float(value))
            elif isinstance(value, str) and value.strip():
                try:
                    duration = max(0, float(value.strip()))
                except Exception:
                    duration = 1
            await asyncio.sleep(duration)
            agent.logger.info(f"Waited {duration} seconds")
            return f"Waited {duration} seconds"
        except Exception as e:
            agent.logger.error(f"WAIT action failed: {e}")
            return f"FAILED: WAIT error - {e}"
    elif action_name == "PRESS ENTER":
        try:
            focused_element = await page.evaluate("""
                () => {
                    const activeElement = document.activeElement;
                    if (activeElement && activeElement !== document.body) {
                        const rect = activeElement.getBoundingClientRect();
                        return {
                            tagName: activeElement.tagName,
                            type: activeElement.type || '',
                            id: activeElement.id || '',
                            className: activeElement.className || '',
                            placeholder: activeElement.placeholder || '',
                            value: activeElement.value || '',
                            center: {
                                x: rect.left + rect.width / 2,
                                y: rect.top + rect.height / 2
                            },
                            rect: {
                                left: rect.left,
                                top: rect.top,
                                width: rect.width,
                                height: rect.height
                            }
                        };
                    }
                    return null;
                }
            """)
            
            if focused_element:
                agent.logger.info(f"Found focused element for PRESS ENTER: {focused_element['tagName']} (type: {focused_element.get('type', 'N/A')})")
                await page.keyboard.press('Enter')
                return f"SUCCESS: Pressed Enter on focused {focused_element['tagName']} element"
            
            search_element = await page.evaluate("""
                () => {
                    const selectors = [
                        'input[type="search"]',
                        'input[type="text"]',
                        'input[placeholder*="search" i]',
                        'input[placeholder*="enter" i]',
                        'input[name*="search" i]',
                        'input[id*="search" i]',
                        'textarea',
                        'input:not([type="button"]):not([type="submit"]):not([type="reset"]):not([type="checkbox"]):not([type="radio"])'
                    ];
                    
                    for (const selector of selectors) {
                        const elements = document.querySelectorAll(selector);
                        for (const element of elements) {
                            const rect = element.getBoundingClientRect();
                            if (rect.width > 0 && rect.height > 0 && 
                                rect.top >= 0 && rect.left >= 0 &&
                                rect.bottom <= window.innerHeight && 
                                rect.right <= window.innerWidth) {
                                return {
                                    tagName: element.tagName,
                                    type: element.type || '',
                                    id: element.id || '',
                                    className: element.className || '',
                                    placeholder: element.placeholder || '',
                                    selector: selector,
                                    center: {
                                        x: rect.left + rect.width / 2,
                                        y: rect.top + rect.height / 2
                                    }
                                };
                            }
                        }
                    }
                    return null;
                }
            """)
            
            if search_element:
                agent.logger.info(f"Found search element for PRESS ENTER: {search_element['tagName']} (selector: {search_element['selector']})")
                await page.mouse.click(search_element['center']['x'], search_element['center']['y'])
                await page.keyboard.press('Enter')
                return f"SUCCESS: Clicked on {search_element['tagName']} element and pressed Enter"
            
            agent.logger.info("No specific element found for PRESS ENTER, pressing Enter globally")
            await page.keyboard.press('Enter')
            return f"SUCCESS: Pressed Enter globally (no specific element found)"
            
        except Exception as e:
            agent.logger.error(f"Enhanced PRESS ENTER failed, falling back to simple Enter: {e}")
            await page.keyboard.press('Enter')
            return f"SUCCESS: Pressed Enter (fallback after error: {e})"
    elif action_name == "DRAG" or action_name == "DRAG_TO":
        drag_failed = False
        failure_reason = ""
        
        if target_coordinates and not target_element:
            if value and "," in value:
                try:
                    target_x, target_y = map(int, value.split(","))
                    target_x, target_y = agent.map_normalized_to_pixels(target_x, target_y)
                    
                    if isinstance(target_coordinates, dict):
                        sx, sy = target_coordinates["x"], target_coordinates["y"]
                        try:
                            if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                                sx, sy = agent.map_normalized_to_pixels(sx, sy)
                            else:
                                sx, sy = agent.normalize_coords(sx, sy, None)
                        except Exception:
                            if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                                sx, sy = agent.map_normalized_to_pixels(sx, sy)
                            else:
                                sx, sy = agent.normalize_coords(sx, sy, None)
                        await agent.page.mouse.move(round(sx), round(sy))
                        await agent.page.mouse.down()
                        await agent.page.mouse.move(round(target_x), round(target_y))
                        await agent.page.mouse.up()
                        
                        agent.logger.info(f"Dragged from ({target_coordinates['x']}, {target_coordinates['y']}) to ({target_x}, {target_y})")
                        return f"Dragged from ({target_coordinates['x']}, {target_coordinates['y']}) to ({target_x}, {target_y})"
                    elif isinstance(target_coordinates, (tuple, list)) and len(target_coordinates) >= 2:
                        sx, sy = target_coordinates[0], target_coordinates[1]
                        try:
                            if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                                sx, sy = agent.map_normalized_to_pixels(sx, sy)
                            else:
                                sx, sy = agent.normalize_coords(sx, sy, None)
                        except Exception:
                            if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                                sx, sy = agent.map_normalized_to_pixels(sx, sy)
                            else:
                                sx, sy = agent.normalize_coords(sx, sy, None)
                        await agent.page.mouse.move(round(sx), round(sy))
                        await agent.page.mouse.down()
                        await agent.page.mouse.move(round(target_x), round(target_y))
                        await agent.page.mouse.up()
                        
                        agent.logger.info(f"Dragged from ({target_coordinates[0]}, {target_coordinates[1]}) to ({target_x}, {target_y})")
                        return f"Dragged from ({target_coordinates[0]}, {target_coordinates[1]}) to ({target_x}, {target_y})"
                    else:
                        agent.logger.error(f"Invalid target_coordinates format for DRAG action: {type(target_coordinates)} - {target_coordinates}")
                        drag_failed = True
                        failure_reason = f"Invalid coordinate format: {type(target_coordinates)}"
                except Exception as e:
                    drag_failed = True
                    failure_reason = f"Coordinate drag failed - {e}"
                    agent.logger.warning(failure_reason)
            else:
                drag_failed = True
                failure_reason = "DRAG_TO requires target coordinates in format 'x,y'"
                agent.logger.error(failure_reason)
        elif selector and value and "," in value:
            try:
                target_x, target_y = map(int, value.split(","))
                target_x, target_y = agent.map_normalized_to_pixels(target_x, target_y)
                
                box = await selector.bounding_box()
                if box:
                    source_x = box['x'] + box['width'] / 2
                    source_y = box['y'] + box['height'] / 2
                    
                    await agent.page.mouse.move(round(source_x), round(source_y))
                    await agent.page.mouse.down()
                    await agent.page.mouse.move(round(target_x), round(target_y))
                    await agent.page.mouse.up()
                    
                    agent.logger.info(f"Dragged element from ({source_x}, {source_y}) to ({target_x}, {target_y}): {element_repr}")
                    return f"Dragged element from ({source_x}, {source_y}) to ({target_x}, {target_y}): {element_repr}"
                else:
                    drag_failed = True
                    failure_reason = "Could not get element bounding box for drag operation"
                    agent.logger.error(failure_reason)
            except Exception as e:
                drag_failed = True
                failure_reason = f"Element drag failed - {e}"
                agent.logger.warning(failure_reason)
        else:
            drag_failed = True
            failure_reason = "No valid source or target for DRAG action"
            agent.logger.error(failure_reason)
        
        if drag_failed:
            return f"FAILED: {failure_reason}"
    elif action_name == "SELECT":
        if not value or value.strip() == "" or value.lower() in ["none", "null"]:
            agent.logger.error(f"❌ [SELECT] Failed | Reason: Invalid value '{value}'")
            return "FAILED: SELECT action requires a valid value to select"
        
        if not field_name:
            field_name = 'unknown'
        agent.logger.info(f"🎯 [SELECT] Starting | Field: '{field_name}' | Option: '{value}' | Coords: {target_coordinates}")
        
        coords = None
        if isinstance(target_coordinates, dict):
            coords = (target_coordinates["x"], target_coordinates["y"])
        elif isinstance(target_coordinates, (tuple, list)) and len(target_coordinates) >= 2:
            coords = (target_coordinates[0], target_coordinates[1])
        
        if coords:
            try:
                nx, ny = coords
                try:
                    if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                        nx, ny = agent.map_normalized_to_pixels(nx, ny)
                    else:
                        nx, ny = agent.normalize_coords(nx, ny, None)
                except Exception:
                    await agent.take_screenshot()
                    try:
                        if getattr(agent, '_current_coordinates_type', 'normalized') == 'normalized':
                            nx, ny = agent.map_normalized_to_pixels(nx, ny)
                        else:
                            nx, ny = agent.normalize_coords(nx, ny, None)
                    except Exception:
                        return f"FAILED: Coordinate normalization failed"
                agent.logger.info(f"🔧 [SELECT L1/2] Method: Pixel+DOM | Coords: ({nx}, {ny})")
                success, message = await dom_select_option(agent.page, (nx, ny), value)
                if success:
                    agent.logger.info(f"✅ [SELECT L1/2] Success | Method: Pixel+DOM | {message}")
                    return f"SUCCESS: {message} (method: pixel+DOM)"
                else:
                    agent.logger.warning(f"⚠️ [SELECT L1/2] Failed | Method: Pixel+DOM | Reason: {message} | Trying Layer 2")
            except Exception as e:
                agent.logger.warning(f"⚠️ [SELECT L1/2] Exception | Method: Pixel+DOM | Error: {str(e)[:100]} | Trying Layer 2")
        else:
            agent.logger.info(f"⚠️ [SELECT L1/2] Skipped | Reason: No coordinates provided | Trying Layer 2")
        
        try:
            agent.logger.info(f"🔧 [SELECT L2/2] Method: Semantic+LLM | Target field: '{field_name}' | Scanning page...")
            
            all_selects = await get_all_select_elements(agent.page)
            
            if not all_selects:
                agent.logger.error(f"❌ [SELECT L2/2] Failed | Reason: No dropdown fields found on page")
                return f"FAILED: No dropdown fields available for selection"
            
            agent.logger.info(f"📋 [SELECT L2/2] Found {len(all_selects)} dropdowns | Asking LLM to find '{field_name}' dropdown...")
            
            chosen_select = await decide_form_element(
                engine=agent.engine,
                inputs=[],
                selects=all_selects,
                action_type="SELECT",
                value=value,
                task=f"Find '{field_name}' dropdown for: {agent.tasks[-1] if agent.tasks else 'Unknown task'}",
                logger=agent.logger
            )
            
            if chosen_select:
                selector = chosen_select.get('selector')
                agent.logger.info(f"📋 [SELECT L2/2] LLM selected | Dropdown: {chosen_select.get('description')} | Selector: {selector}")
                
                success, message = await dom_select_by_selector(agent.page, selector, value)
                
                if success:
                    agent.logger.info(f"✅ [SELECT L2/2] Success | Method: Semantic+LLM | {message}")
                    return f"SUCCESS: {message} (method: semantic+LLM)"
                else:
                    agent.logger.error(f"❌ [SELECT L2/2] Failed | Method: Semantic+LLM | Reason: {message}")
                    return f"FAILED: {message}"
            else:
                agent.logger.error(f"❌ [SELECT L2/2] Failed | Reason: LLM could not select appropriate dropdown")
                return f"FAILED: No appropriate dropdown found for '{value}'"
        except Exception as e:
            agent.logger.error(f"❌ [SELECT] Failed | All methods exhausted | Error: {str(e)[:200]}")
            agent.logger.debug(traceback.format_exc())
            return f"FAILED: All SELECT methods failed - {e}"
        
    elif action_name == "TERMINATE":
        if await agent._is_page_blocked_or_blank():
            agent.complete_flag = True
            agent.logger.info("Page is blocked/error - allowing termination without task completion check")
            return "Page blocked or error detected. Task cannot be completed. Terminating..."
        
        agent.complete_flag = True
        agent.logger.info("Agent requested termination. Terminating...")
        return "Agent requested termination. Terminating..."
    elif action_name in ["NONE"]:
        agent.logger.info("No action necessary at this stage. Skipped")
        return "No action necessary at this stage. Skipped"
    elif action_name in ["SAY"]:
        say_text = (value or "").strip()
        if not say_text or say_text.lower() == "none":
            agent.logger.warning("SAY action received with empty/None VALUE. Continuing execution.")
            return "SAY action received with empty/None VALUE. Continuing execution."
        
        agent.logger.info(f"SAY (Chain of Thought): {say_text}")
        return f"SAY (thinking step): {say_text}"
    elif action_name == "GUI_GROUNDING":
        agent.logger.info("GUI_GROUNDING disabled: use main model actions with normalized coordinates")
        return "FAILED: GUI_GROUNDING disabled"
    else:
        agent.logger.error(f"Unsupported or improperly specified action: {action_name}")
        return f"FAILED: Unsupported action - {action_name}"

    if action_name in agent.no_element_op and target_element is None:
        new_action = action_name
    else:
        if selector == "pixel_coordinates":
            new_action = element_repr + " -> " + action_name
        else:
            if target_element and isinstance(target_element, dict):
                new_action = "[" + target_element.get('tag_with_role', 'unknown') + "]" + " "
                new_action += target_element.get('description', 'unknown element') + " -> " + action_name
            else:
                new_action = "[unknown element] -> " + action_name
    if action_name in agent.with_value_op:
        new_action += ": " + value

    return new_action

def compose_action_description(action, value, field, element_desc, coords=None):
    a = action or ''
    v = value or ''
    f = field or ''
    d = element_desc or ''
    if a == 'CLICK':
        return "Clicked"
    if a == 'HOVER':
        return f"Hovered on {d}" if d else "Hovered"
    if a == 'TYPE':
        if f:
            return f"Typed '{v}' in {f}"
        if d:
            return f"Typed '{v}' in {d}"
        return f"Typed '{v}'"
    if a == 'SELECT':
        if f:
            return f"Selected '{v}' from {f}"
        if d:
            return f"Selected '{v}' from {d}"
        return f"Selected '{v}'"
    if a == 'PRESS ENTER':
        return "Pressed Enter"
    if a == 'SCROLL UP':
        return "Scrolled up"
    if a == 'SCROLL DOWN':
        return "Scrolled down"
    if a == 'SCROLL TOP':
        return "Scrolled to top"
    if a == 'SCROLL BOTTOM':
        return "Scrolled to bottom"
    if a == 'GOTO':
        return f"Navigated to {v}" if v else "Navigated"
    if a == 'WAIT':
        return f"Waited {v} seconds" if v else "Waited"
    if a == 'KEYBOARD':
        if v and v.upper() == 'CLEAR':
            return "Cleared field via keyboard"
        return f"Typed '{v}' via keyboard" if v else "Typed via keyboard"
    if a == 'NEW TAB':
        return "Opened new tab"
    if a == 'CLOSE TAB':
        return "Closed tab"
    if a == 'GO BACK':
        return "Went back"
    if a == 'GO FORWARD':
        return "Went forward"
    if a == 'TERMINATE':
        return "Terminated"
    return a or "Action"

async def execute(agent, prediction_dict):
    """
    Execute the predicted action on the webpage.
    """

    # Enhanced safety checks with detailed logging
    if prediction_dict is None:
        agent.logger.warning("=== PREDICTION DICT IS NONE ===")
        agent.logger.warning("Received None prediction_dict, marking task as complete")
        agent.logger.warning("This may indicate max operations reached or prediction failure")
        agent.complete_flag = True
        return

    # Validate prediction_dict structure
    if not isinstance(prediction_dict, dict):
        agent.logger.error("=== INVALID PREDICTION DICT TYPE ===")
        agent.logger.error(f"Expected dict, got {type(prediction_dict)}: {prediction_dict}")
        agent.logger.error("Marking task as complete due to invalid prediction structure")
        agent.complete_flag = True
        return

    # Validate required keys in prediction_dict
    # Check for required keys with detailed logging
    required_keys = ["element", "action", "value"]
    missing_keys = [key for key in required_keys if key not in prediction_dict]
    if missing_keys:
        agent.logger.error("=== MISSING REQUIRED KEYS IN PREDICTION DICT ===")
        agent.logger.error(f"Missing keys: {missing_keys}")
        agent.logger.error(f"Available keys: {list(prediction_dict.keys())}")
        agent.logger.error(f"Full prediction_dict: {prediction_dict}")
        agent.logger.error("Using default values for missing keys")
        
        # Use safe defaults instead of failing
        for key in missing_keys:
            if key == "element":
                prediction_dict[key] = None
            elif key == "action":
                prediction_dict[key] = "NONE"
            elif key == "value":
                prediction_dict[key] = ""

    # Safe dictionary access with enhanced logging and type checking
    pred_element = prediction_dict.get("element")
    pred_action = prediction_dict.get("action", "NONE")
    pred_value = prediction_dict.get("value", "")
    pred_coordinate = None
    pred_element_description = None
    pred_coordinate_type = "normalized"
    
    # Ensure pred_action is a string and not None
    if pred_action is None:
        agent.logger.warning("pred_action is None, setting to 'NONE'")
        pred_action = "NONE"
    elif not isinstance(pred_action, str):
        agent.logger.warning(f"pred_action is not a string: {type(pred_action)}, converting to string")
        pred_action = str(pred_action)
    
    # Ensure pred_value is not None
    if pred_value is None:
        agent.logger.warning("pred_value is None, setting to empty string")
        pred_value = ""
    
    pred_element_description = (
        prediction_dict.get("action_description")
        or prediction_dict.get("description")
        or ""
    )
    # Drag endpoint descriptions (new format)
    drag_source_desc = prediction_dict.get("drag_source_desc") or prediction_dict.get("source_desc")
    drag_target_desc = prediction_dict.get("drag_target_desc") or prediction_dict.get("target_desc")
    
    # Use coordinates provided by prediction when available
    if "coordinates" in prediction_dict:
        pred_coordinate = prediction_dict["coordinates"]
        pred_coordinate_type = prediction_dict.get("coordinates_type") or "normalized"
        try:
            if isinstance(pred_coordinate, (list, tuple)) and len(pred_coordinate) >= 2:
                pred_coordinate = (float(pred_coordinate[0]), float(pred_coordinate[1]))
        except Exception as e:
            agent.logger.warning(f"Invalid coordinates format: {e}")

    # Log execution start
    agent.logger.info("=== EXECUTING PREDICTED ACTION ===")
    agent.logger.info(f"Step: {agent.time_step}")
    agent.logger.info(f"Predicted Action: {pred_action}")
    agent.logger.info(f"Predicted Value: {pred_value}")
    agent.logger.info(f"Element Description: {pred_element_description}")
    agent.logger.info(f"Element: {pred_element}")
    agent.logger.info(f"Coordinates: {pred_coordinate}")

    # page_state_before = await agent._capture_page_state()

    try:
        # Special handling for TYPE/SELECT without element or coordinates
        # Extract available fields and let LLM choose which one to use
        if pred_action == "TYPE" and pred_element is None and pred_coordinate is None:
            agent.logger.info("TYPE action without element/coordinates - extracting available fields")
            available_fields = await extract_typeable_elements(agent.page, logger=agent.logger)
            
            if available_fields:
                agent.logger.info(f"Found {len(available_fields)} typeable fields")
                # Ask LLM to choose which field
                chosen_field = await choose_field_with_llm(
                    agent.engine,
                    agent.model,
                    agent.logger,
                    available_fields, 
                    "TYPE", 
                    pred_value, 
                    agent.tasks[-1] if agent.tasks else "Unknown task"
                )
                
                if chosen_field:
                    # Use the chosen field's coordinates
                    pred_coordinate = (chosen_field['center']['x'], chosen_field['center']['y'])
                    pred_coordinate_type = 'pixel'
                    agent.logger.info(f"✅ Selected field for TYPE: {chosen_field.get('label') or chosen_field.get('placeholder') or 'Unnamed'}")
                    agent.logger.info(f"   Coordinates: {pred_coordinate}")
                else:
                    agent.logger.warning("LLM failed to choose a field")
                    pred_action = "NONE"
            else:
                agent.logger.warning("No typeable fields found on page")
                pred_action = "NONE"
        
        elif pred_action == "SELECT" and pred_element is None and pred_coordinate is None:
            agent.logger.info("SELECT action without element/coordinates - extracting available dropdowns")
            available_dropdowns = await extract_selectable_elements(agent.page, logger=agent.logger)
            
            if available_dropdowns:
                agent.logger.info(f"Found {len(available_dropdowns)} selectable dropdowns")
                # Ask LLM to choose which dropdown
                chosen_dropdown = await choose_field_with_llm(
                    agent.engine,
                    agent.model,
                    agent.logger,
                    available_dropdowns,
                    "SELECT",
                    pred_value,
                    agent.tasks[-1] if agent.tasks else "Unknown task"
                )
                
                if chosen_dropdown:
                    # Create a proper element dict with selector for SELECT action
                    pred_element = {
                        'selector': chosen_dropdown.get('selector'),
                        'description': chosen_dropdown.get('label') or chosen_dropdown.get('name') or 'Selected dropdown',
                        'center_point': (chosen_dropdown['center']['x'], chosen_dropdown['center']['y']),
                        'tag_name': 'select'
                    }
                    pred_coordinate = (chosen_dropdown['center']['x'], chosen_dropdown['center']['y'])
                    pred_coordinate_type = 'pixel'
                    agent.logger.info(f"✅ Selected dropdown for SELECT: {chosen_dropdown.get('label') or 'Unnamed'}")
                    agent.logger.info(f"   Selector: {chosen_dropdown.get('selector')}")
                    agent.logger.info(f"   Coordinates: {pred_coordinate}")
                else:
                    agent.logger.warning("LLM failed to choose a dropdown")
                    pred_action = "NONE"
            else:
                agent.logger.warning("No selectable dropdowns found on page")
                pred_action = "NONE"
        
        # In the unified system, actions can use coordinates instead of elements
        # Only convert to NONE if BOTH element and coordinates are missing (after trying to find them above)
        elif pred_action == "DRAG" and pred_coordinate is None and (not pred_value or "," not in str(pred_value)):
            agent.logger.warning("DRAG requires source and target coordinates; converting to NONE")
            pred_action = "NONE"
        elif (pred_action not in agent.no_element_op) and pred_element == None and pred_coordinate == None:
            if pred_action == "CLICK":
                target_text = (pred_value or '').strip() or (pred_element_description or '').strip()
                if target_text:
                    found = await find_click_target_by_text(agent.page, target_text, logger=agent.logger)
                    if found:
                        pred_element, pred_coordinate, pred_coordinate_type = found
                    else:
                        agent.logger.warning(f"Action {pred_action} requires element or coordinates, but both are None. Converting to NONE.")
                        pred_action = "NONE"
                else:
                    agent.logger.warning(f"Action {pred_action} requires element or coordinates, but both are None. Converting to NONE.")
                    pred_action = "NONE"
            else:
                agent.logger.warning(f"Action {pred_action} requires element or coordinates, but both are None. Converting to NONE.")
                pred_action = "NONE"
            
        # Extract field parameter for TYPE/SELECT actions
        pred_field = prediction_dict.get('field', None)
        try:
            agent._pending_hit_test_coords = None
            if pred_action == "CLICK" and isinstance(pred_coordinate, (list, tuple)) and len(pred_coordinate) >= 2:
                cx, cy = pred_coordinate[0], pred_coordinate[1]
                if pred_coordinate_type == "normalized":
                    px, py = agent.map_normalized_to_pixels(cx, cy)
                else:
                    px, py = agent.normalize_coords(cx, cy, None)
                agent._pending_hit_test_coords = (int(px), int(py))
        except Exception:
            agent._pending_hit_test_coords = None
        
        # Capture page state before action
        try:
            state_before = await agent._capture_page_state()
        except Exception:
            state_before = {"error": "capture_failed_before"}

        agent._current_coordinates_type = pred_coordinate_type
        new_action = await perform_action(
            agent,
            pred_element,
            pred_action,
            pred_value,
            pred_coordinate,
            pred_element_description,
            pred_field,
            prediction_dict.get('action_description'),
            clear_first=prediction_dict.get('clear_first', True) if pred_action in ("TYPE", "KEYBOARD") else True,
            press_enter_after=prediction_dict.get('press_enter_after', False) if pred_action == "TYPE" else False
        )
        
        # Add protection against None return values from perform_action
        if new_action is None:
            agent.logger.warning("=== PERFORM_ACTION RETURNED NONE ===")
            agent.logger.warning(f"Action: {pred_action}, Element: {pred_element}, Value: {pred_value}")
            new_action = f"Action {pred_action} completed but returned None - this should not happen"
            agent.logger.warning(f"Using fallback action description: {new_action}")
        elif not isinstance(new_action, str):
            agent.logger.warning(f"=== PERFORM_ACTION RETURNED NON-STRING: {type(new_action)} ===")
            agent.logger.warning(f"Value: {new_action}")
            new_action = str(new_action) if new_action is not None else f"Action {pred_action} completed"
            agent.logger.warning(f"Converted to string: {new_action}")
        
        # Wait for potential page changes after action
        await asyncio.sleep(1)
        try:
            state_after = await agent._capture_page_state()
        except Exception:
            state_after = {"error": "capture_failed_after"}
        try:
            new_action = compose_action_description(
                pred_action,
                pred_value,
                pred_field,
                pred_element_description,
                pred_coordinate,
            )
        except Exception:
            pass
        
        # Detect whether the action had any effect on the page
        try:
            agent._last_changes_detected = []
            action_success = await agent._detect_page_state_change(state_before, state_after, pred_action)
        except Exception:
            action_success = True
        
        # Log action execution and annotate screenshot if no-effect with coordinates
        agent.logger.info(f"Action executed: {new_action}")
        labeled_screenshot = None
        if not action_success:
            try:
                # Use last click viewport coords or predicted coordinate for labeling
                coords_to_mark = None
                source_type = None
                if agent.last_click_viewport_coords:
                    coords_to_mark = agent.last_click_viewport_coords
                    source_type = 'pixel'
                elif isinstance(pred_coordinate, (list, tuple)) and len(pred_coordinate) >= 2:
                    coords_to_mark = (int(pred_coordinate[0]), int(pred_coordinate[1]))
                    source_type = pred_coordinate_type or 'normalized'
                # Take and annotate current screenshot when coordinates are available
                await agent.take_screenshot()
                if coords_to_mark:
                    cx, cy = coords_to_mark
                    if source_type == 'normalized':
                        nx, ny = agent.map_normalized_to_pixels(cx, cy)
                        agent.last_click_viewport_coords = (int(nx), int(ny))
                    else:
                        agent.last_click_viewport_coords = (int(cx), int(cy))
                try:
                    await agent._annotate_current_screenshot()
                    labeled_screenshot = os.path.join(agent.main_path, 'screenshots', f'screen_{agent.time_step}_labeled.png')
                    agent.logger.info(f"No-effect detected; labeled screenshot: {labeled_screenshot}")
                except Exception as ann_e:
                    agent.logger.warning(f"Failed to annotate screenshot after no-effect: {ann_e}")
            except Exception as ss_e:
                agent.logger.warning(f"Failed to capture/label screenshot after no-effect: {ss_e}")
        
        # Create enhanced action record with response details and page content summary
        enhanced_action = {
            "step": agent.time_step,
            "action": pred_action,
            "action_description": prediction_dict.get("action_description") or new_action,
            "action_generation_response": prediction_dict.get("action_generation", ""),
            "action_grounding_response": prediction_dict.get("action_grounding", ""),
            "predicted_action": pred_action,
            "predicted_value": pred_value,
            "element_description": pred_element_description if pred_element_description else (pred_element.get('description', '') if pred_element else ''),
            "coordinates": pred_coordinate,
            "element_center": (pred_element.get('center_point') if pred_element else None),
            "element_box": (pred_element.get('box') if pred_element else None),
            "http_response": agent.session_control.get('last_response', {}) if agent.session_control and isinstance(agent.session_control, dict) else {},
            "success": action_success,
            "error": None,
            "page_content_summary": "Page content summary unavailable",
            "next": "",
            "labeled_screenshot": labeled_screenshot,
            "changes_detected": getattr(agent, "_last_changes_detected", None)
        }
        
        # Log enhanced action details
        agent.logger.info("Enhanced action record created:")
        try:
            http_resp = enhanced_action['http_response'] if isinstance(enhanced_action.get('http_response'), dict) else {}
            agent.logger.info(f"  HTTP Response: url={http_resp.get('url')}, status={http_resp.get('status')}")
        except Exception:
            pass
        
        agent.taken_actions.append(enhanced_action)
        # Also add to action_history for failure analysis
        agent.action_history.append(enhanced_action)
        agent.logger.info(f"Total actions in history: {len(agent.taken_actions)}")
        
        # Manage action history: compress and limit length
        agent._manage_action_history()
        
        # Update checklist based on action execution
        await agent._update_checklist_after_action(enhanced_action)
        
        agent.logger.info("=== ACTION EXECUTION COMPLETE ===")
        if pred_action != "NONE":
            agent.valid_op += 1
            agent.continuous_no_op = 0
        else:
            agent.continuous_no_op += 1
        try:
            agent._pending_hit_test_coords = None
        except Exception:
            pass
        if agent.config["basic"].get("crawler_mode", False) is True:
            await agent.stop_playwright_tracing()
            await agent.save_traces()

        return 0
    except Exception as e:
        # Enhanced error handling with checklist update
        agent.logger.error(f"Action execution failed: {e}")
        
        try:
            base_desc = compose_action_description(
                pred_action,
                pred_value,
                pred_field,
                pred_element_description,
                pred_coordinate,
            )
            new_action = f"{base_desc} - failed: {e}"
        except Exception:
            new_action = f"Failed to perform {pred_action} with value '{pred_value}': {e}"

        # Create enhanced action record for failed action
        enhanced_action = {
            "step": agent.time_step,
            "action": pred_action,
            "action_description": prediction_dict.get("action_description") or new_action,
            "action_generation_response": prediction_dict.get("action_generation", ""),
            "action_grounding_response": prediction_dict.get("action_grounding", ""),
            "predicted_action": pred_action,
            "predicted_value": pred_value,
            "element_description": pred_element_description if pred_element_description else (pred_element.get('description', '') if pred_element else ''),
            "coordinates": pred_coordinate,
            "element_center": (pred_element.get('center_point') if pred_element else None),
            "element_box": (pred_element.get('box') if pred_element else None),
            "http_response": {},
            "success": False,
            "error": str(e),
            "page_content_summary": "Operation failed, unable to get page content",
            "next": "",
            "changes_detected": getattr(agent, "_last_changes_detected", None)
        }
        
        agent.taken_actions.append(enhanced_action)
        agent.action_history.append(enhanced_action)
        
        # Manage action history: compress and limit length
        agent._manage_action_history()
        
        # Update checklist for failed action
        await agent._update_checklist_after_action(enhanced_action)
        
        traceback_info = traceback.format_exc()
        error_message = f"Error executing action {pred_action}: {str(e)}"
        print(traceback_info)
        # exit()
        error_message_with_traceback = f"{error_message}\n\nTraceback:\n{traceback_info}"

        # Log error details
        agent.logger.error("=== ACTION EXECUTION ERROR ===")
        agent.logger.error(f"Failed Action: {pred_action}")
        agent.logger.error(f"Target Element: {pred_element}")
        agent.logger.error(f"Value: {pred_value}")
        agent.logger.error(f"Error: {str(e)}")
        agent.logger.error(f"Traceback: {traceback_info}")

        # CRITICAL FIX: Take a fresh screenshot after action failure
        # This ensures LLM gets the current page state for next analysis
        try:
            await agent.take_screenshot()
            agent.logger.info(f"Post-error screenshot saved: {agent.screenshot_path}")
        except Exception as screenshot_error:
            agent.logger.error(f"Failed to take post-error screenshot: {screenshot_error}")

        # Create enhanced action record for error case
        enhanced_action = {
            "step": agent.time_step,
            "action_description": new_action,
            "action_generation_response": prediction_dict.get("action_generation", ""),
            "action_grounding_response": prediction_dict.get("action_grounding", ""),
            "predicted_action": pred_action,
            "predicted_value": pred_value,
            "element_description": pred_element_description if pred_element_description else (pred_element.get('description', '') if pred_element else ''),
            "success": False,
            "error": str(e),
            "coordinates": pred_coordinate,
            "element_center": (pred_element.get('center_point') if pred_element else None),
            "element_box": (pred_element.get('box') if pred_element else None),
            "http_response": agent.session_control.get('last_response', {}) if agent.session_control and isinstance(agent.session_control, dict) else {},
            "page_content_summary": "Operation failed, unable to get page content"
        }

        agent.logger.info(new_action)
        agent.logger.info("Enhanced error action record created:")
        agent.logger.info(f"  Error: {enhanced_action['error']}")
        agent.logger.info(f"  HTTP Response: {enhanced_action['http_response']}")
        agent.taken_actions.append(enhanced_action)
        # Also add to action_history for failure analysis
        enhanced_action['failure_type'] = 'execution_error'  # Set failure type for analysis
        agent.action_history.append(enhanced_action)
        agent.logger.info(f"Total actions in history: {len(agent.taken_actions)}")
        
        # Manage action history: compress and limit length
        agent._manage_action_history()
        
        # Action failure - let LLM re-evaluate in next iteration (none behavior)
        agent.logger.info("Action failure detected - allowing LLM to re-evaluate (none option behavior)")
        agent.logger.error("=== ACTION EXECUTION ERROR COMPLETE ===")
        agent.continuous_no_op += 1
        return 1
