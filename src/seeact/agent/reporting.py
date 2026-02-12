# -*- coding: utf-8 -*-
import os
import json
import logging
import datetime
import tempfile
from playwright.async_api import Locator
from seeact.browser.helper import saveconfig

def _compress_text(text: str, max_length: int) -> str:
    """Compress text by truncating and adding ellipsis if needed."""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def _compress_url(url: str) -> str:
    """Compress URL by keeping only the domain and path structure."""
    if not url or url == 'Unknown':
        return url
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.netloc:
            # Keep domain and first part of path
            path_parts = parsed.path.split('/')[:3]  # Keep first 2 path segments
            compressed_path = '/'.join(path_parts)
            if len(parsed.path.split('/')) > 3:
                compressed_path += "/..."
            return f"{parsed.netloc}{compressed_path}"
        return url[:50] + "..." if len(url) > 50 else url
    except:
        return url[:50] + "..." if len(url) > 50 else url

def generate_comprehensive_action_summary(taken_actions, predictions=[], reflection_history=None, max_actions=20, compress_old=True) -> str:
    """
    Generate a comprehensive but compressed summary of actions taken for evaluation context.
    Includes compressed historical information with intelligent truncation to reduce token usage.
    """
    if not taken_actions:
        return "No actions taken."
    
    summary_lines = []
    
    # Handle action history compression
    total_actions = len(taken_actions)
    if total_actions > max_actions and compress_old:
        # Show compressed summary for older actions
        older_count = total_actions - max_actions
        summary_lines.append(f"=== ACTION HISTORY ({total_actions} total, showing recent {max_actions}) ===")
        summary_lines.append(f"[{older_count} earlier actions compressed]")
        actions_to_process = taken_actions[-max_actions:]
        start_index = older_count + 1
    else:
        summary_lines.append("=== COMPLETE ACTION HISTORY ===")
        actions_to_process = taken_actions
        start_index = 1
    
    # Process actions with compression
    for i, action in enumerate(actions_to_process, start_index):
        if isinstance(action, dict):
            # Basic action information
            action_type = action.get('predicted_action', 'UNKNOWN')
            element_desc = _compress_text(action.get('element_description', ''), 100)
            action_value = _compress_text(action.get('predicted_value', ''), 50)
            success = action.get('success', True)
            error = _compress_text(action.get('error', ''), 150)
            
            # Get corresponding prediction for thought process
            thought = ''
            action_generation = ''
            action_grounding = ''
            if (i - start_index + 1) <= len(predictions):
                pred = predictions[i - start_index]
                thought = pred.get('action_generation', {}).get('thought', '') if isinstance(pred.get('action_generation'), dict) else ''
                action_generation = pred.get('action_generation', '')
                action_grounding = pred.get('action_grounding', '')
            
            # Compress thought and grounding information
            if thought:
                thought = _compress_text(thought, 200)
            if action_generation and isinstance(action_generation, str):
                action_generation = _compress_text(action_generation, 300)
            if action_grounding:
                action_grounding = _compress_text(str(action_grounding), 150)
            
            # Get page context (compressed)
            page_url = _compress_url(action.get('page_url', 'Unknown'))
            page_title = _compress_text(action.get('page_title', 'Unknown'), 50)
            
            status = "SUCCESS" if success else "FAILED"
            
            summary_lines.append(f"\n--- Step {i}: {status} ---")
            summary_lines.append(f"Action: {action_type}")
            if action_value:
                summary_lines.append(f"Value: {action_value}")
            if element_desc:
                summary_lines.append(f"Target Element: {element_desc}")
            summary_lines.append(f"Page: {page_title} ({page_url})")
            
            # Include compressed thought process and reasoning
            if thought:
                summary_lines.append(f"Reasoning: {thought}")
            elif action_generation:
                # Extract thought from action generation if available
                if "thought:" in action_generation.lower():
                    thought_part = action_generation.split("thought:")[-1].split("action:")[0].strip()
                    if thought_part:
                        summary_lines.append(f"Reasoning: {_compress_text(thought_part, 200)}")
            
            # Include compressed grounding information
            if action_grounding:
                summary_lines.append(f"Element Selection: {action_grounding}")
            
            # Include compressed error information if failed
            if error:
                summary_lines.append(f"Error: {error}")
            
            # Include compressed additional context
            if action.get('action_description'):
                compressed_desc = _compress_text(action.get('action_description'), 100)
                summary_lines.append(f"Description: {compressed_desc}")
    
    # Add overall execution context
    summary_lines.append(f"\n=== EXECUTION SUMMARY ===")
    summary_lines.append(f"Total Steps: {total_actions}")
    successful_actions = sum(1 for action in taken_actions if isinstance(action, dict) and action.get('success', True))
    summary_lines.append(f"Successful Actions: {successful_actions}/{total_actions}")
    
    # Add compressed reflection history if available
    if reflection_history:
        summary_lines.append(f"\n=== RECENT REFLECTIONS ===")
        for i, reflection in enumerate(reflection_history[-3:], 1):  # Last 3 reflections
            compressed_reflection = _compress_text(reflection, 200)
            summary_lines.append(f"Reflection {i}: {compressed_reflection}")
    
    return "\n".join(summary_lines)

def generate_recent_action_summary(taken_actions) -> str:
    """
    Generate a simplified summary of recent actions for evaluation.
    Only includes essential information: action type, target, and result.
    """
    if not taken_actions:
        return "No actions taken."
    
    # Get last 10 actions or all if fewer
    recent_actions = taken_actions[-10:]
    summary_lines = []
    
    for i, action in enumerate(recent_actions, 1):
        if isinstance(action, dict):
            action_type = action.get('predicted_action', 'UNKNOWN')
            element_desc = action.get('element_description', '')
            action_value = action.get('predicted_value', '')
            success = action.get('success', True)
            
            status = "✓" if success else "✗"
            
            # Simple format: Step X: [Status] ACTION on TARGET with VALUE
            line = f"Step {i}: [{status}] {action_type}"
            if element_desc:
                line += f" on {element_desc}"
            if action_value:
                line += f" with '{action_value}'"
            
            summary_lines.append(line)
    
    return "\n".join(summary_lines)

def generate_action_description(parsed_action, logger=None):
    """
    Generate a human-readable description from parsed action.
    """
    try:
        action = parsed_action.get('action', 'UNKNOWN')
        coords = parsed_action.get('coordinates', [])
        value = parsed_action.get('value', '')
        field = parsed_action.get('field', '')
        
        # Format based on action type
        if action == "CLICK":
            if coords:
                return f"Click at ({coords[0]}, {coords[1]})"
            return "Click"
        
        elif action == "TYPE":
            desc = f"Type '{value}'"
            if field:
                desc += f" in {field} field"
            return desc
        
        elif action == "SELECT":
            desc = f"Select '{value}'"
            if field:
                desc += f" from {field} dropdown"
            return desc
        
        elif action == "HOVER":
            if coords:
                return f"Hover at ({coords[0]}, {coords[1]})"
            return "Hover"
        
        elif action == "DRAG":
            if coords and len(coords) >= 4:
                return f"Drag from ({coords[0]}, {coords[1]}) to ({coords[2]}, {coords[3]})"
            return "Drag"
        
        elif action == "SCROLL UP":
            return "Scroll up"
        
        elif action == "SCROLL DOWN":
            return "Scroll down"
        
        elif action == "SCROLL TOP":
            return "Scroll to top"
        
        elif action == "SCROLL BOTTOM":
            return "Scroll to bottom"
        
        elif action == "PRESS ENTER":
            return "Press Enter"
        
        elif action == "GOTO":
            return f"Navigate to {value}"
        
        elif action == "NEW TAB":
            return "Open new tab"
        
        elif action == "CLOSE TAB":
            return "Close tab"
        
        elif action == "GO BACK":
            return "Go to previous tab"
        
        elif action == "WAIT":
            return f"Wait {value} seconds"
        
        elif action == "TERMINATE":
            return f"Terminate: {value}" if value else "Terminate task"
        
        else:
            return f"{action}" + (f": {value}" if value else "")
            
    except Exception as e:
        if logger:
            logger.error(f"Failed to generate action description: {e}")
        return f"{parsed_action.get('action', 'UNKNOWN')} action"

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

def emergency_save(task_id, taken_actions, error_info="Unknown error", logger=None):
    """
    Emergency save mechanism when normal save operations fail.
    Saves minimal data to ensure no information is lost.
    """
    try:
        # Create emergency save directory
        emergency_dir = os.path.join(tempfile.gettempdir(), f"seeact_emergency_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(emergency_dir, exist_ok=True)
        
        # Save minimal result data
        emergency_result = {
            "task_id": task_id or 'unknown',
            "emergency_save": True,
            "error_info": error_info,
            "timestamp": datetime.datetime.now().isoformat(),
            "num_actions": len(taken_actions or []),
            "success_or_not": "error",
            "final_result_response": "Task execution encountered an unexpected error and was saved in emergency mode. Please check the action summary for task progress."
        }
        
        # Try to save action history if available
        try:
            if taken_actions:
                emergency_result["action_summary"] = []
                for i, action in enumerate(taken_actions[-5:]):  # Save last 5 actions
                    if isinstance(action, dict):
                        emergency_result["action_summary"].append({
                            "step": action.get('step', i),
                            "action": action.get('predicted_action', 'unknown'),
                            "description": action.get('action_description', '')[:100]  # Truncate long descriptions
                        })
                    else:
                        emergency_result["action_summary"].append(str(action)[:100])
        except Exception:
            emergency_result["action_summary"] = "Failed to extract action summary"
        
        # Save emergency result
        emergency_file = os.path.join(emergency_dir, 'emergency_result.json')
        with open(emergency_file, 'w', encoding='utf-8') as f:
            json.dump(emergency_result, f, indent=4)
        
        # Log emergency save location
        msg = f"EMERGENCY SAVE: Data saved to {emergency_file}"
        if logger:
            logger.error(msg)
        print(msg)
        
        return emergency_file
        
    except Exception as emergency_e:
        # If even emergency save fails, log to console
        error_msg = f"CRITICAL: Emergency save failed: {emergency_e}. Original error: {error_info}"
        if logger:
            logger.critical(error_msg)
        print(error_msg)
        return None

def save_results(main_path, task_id, final_json, taken_actions, config, logger, llm_io_records=None):
    def locator_serializer(obj):
        """Convert non-serializable objects to a serializable format."""
        if isinstance(obj, Locator):
            return str(obj)
        try:
            return str(obj)
        except:
            return f"<Non-serializable object: {type(obj).__name__}>"

    def _simplify_messages(msgs):
        result = {"system": [], "user": [], "assistant": []}
        try:
            for m in msgs or []:
                role = m.get("role", "user")
                content = m.get("content", [])
                texts = []
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            texts.append(str(c.get("text", "")))
                elif isinstance(content, str):
                    texts.append(content)
                if role in result:
                    result[role].append("\n".join(texts))
            for k in result:
                result[k] = "\n\n".join([t for t in result[k] if t])
        except Exception:
            return {"system": "", "user": "", "assistant": ""}
        return result

    def _format_llm_records(records):
        formatted = []
        try:
            for r in records or []:
                msgs = r.get("messages")
                simplified = _simplify_messages(msgs) if msgs else {"system": "", "user": "", "assistant": ""}
                formatted.append({
                    "timestamp": r.get("timestamp"),
                    "model": r.get("model"),
                    "turn_number": r.get("turn_number", 0),
                    "input": simplified,
                    "images": r.get("image_paths") or ([r.get("image_path")] if r.get("image_path") else []),
                    "output": r.get("output"),
                    "context": r.get("context")
                })
        except Exception:
            return []
        return formatted

    # Save all predictions
    try:
        with open(os.path.join(main_path, 'all_predictions.json'), 'w', encoding='utf-8') as f:
            try:
                filtered_records = [r for r in (llm_io_records or []) if r.get('task_id') == task_id]
            except Exception:
                filtered_records = []
            formatted_records = _format_llm_records(filtered_records)
            json.dump(formatted_records, f, default=locator_serializer, indent=4)
        if logger:
            logger.info("Successfully saved all_predictions.json")
    except Exception as e:
        if logger:
            logger.error(f"Failed to save all_predictions.json: {e}")
        try:
            with open(os.path.join(main_path, 'all_predictions.json'), 'w', encoding='utf-8') as f:
                json.dump({"error": f"Failed to save predictions: {str(e)}"}, f, indent=4)
        except Exception as fallback_e:
            if logger:
                logger.error(f"Failed to save fallback all_predictions.json: {fallback_e}")

    # Save result.json
    try:
        with open(os.path.join(main_path, 'result.json'), 'w', encoding='utf-8') as file:
            json.dump(final_json, file, indent=4)
        if logger:
            logger.info("Successfully saved result.json")
    except Exception as e:
        if logger:
            logger.error(f"Failed to save result.json: {e}")
        # Try to save a minimal version
        try:
            minimal_result = {
                "task_id": task_id or 'demo_task',
                "success_or_not": "error",
                "final_result_response": "Task execution completed but unable to save detailed results.",
                "num_step": len(taken_actions) if taken_actions else 0,
                "error": str(e)
            }
            with open(os.path.join(main_path, 'result.json'), 'w', encoding='utf-8') as file:
                json.dump(minimal_result, file, indent=4)
            if logger:
                logger.info("Successfully saved minimal result.json")
        except Exception as fallback_e:
            if logger:
                logger.error(f"Failed to save minimal result.json: {fallback_e}")

    # Save config
    try:
        saveconfig(config, os.path.join(main_path, 'config.toml'))
        if logger:
            logger.info("Successfully saved config.toml")
    except Exception as e:
        if logger:
            logger.error(f"Failed to save config.toml: {e}")

    if logger:
        logger.info("Agent stopped - all save operations attempted.")
