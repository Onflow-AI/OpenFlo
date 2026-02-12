# -*- coding: utf-8 -*-
import re
import os
import logging
from seeact.prompts.utils import generate_action_journey_summary
from seeact.agent.reporting import generate_comprehensive_action_summary
from seeact.prompts.templates import (
    build_termination_prompt,
    build_blocking_analysis_prompt,
    build_completion_verification_prompt,
    build_description_prompt,
    build_evaluation_prompt
)

async def should_terminate_intelligently(agent) -> bool:
    """
    Ultra-conservative intelligent termination - heavily biased toward continuation.
    Only terminates when Agent explicitly signals completion or in extreme failure cases.
    """
    # Check basic termination conditions first
    if agent.complete_flag:
        return True
    
    # Ultra-conservative - require many more actions before considering any termination
    if len(agent.taken_actions) < 25:
        return False
    
    # Only use LLM termination analysis after extensive execution
    if len(agent.taken_actions) >= 30:
        return await llm_should_terminate(agent)
    
    # Enhanced failure pattern detection - extremely conservative thresholds
    if len(agent.action_history) >= 25:
        recent_failures = sum(
            1 for action in agent.action_history[-15:] 
            if not action.get('success', True)
        )
        # Require overwhelming failure rate before considering termination
        if recent_failures >= 12:
            agent.logger.warning("Terminating due to overwhelming recent failures")
            agent.complete_flag = True
            return True
    
    return False

async def llm_should_terminate(agent) -> bool:
    """
    Advanced semantic-aware termination analysis using hierarchical goal understanding.
    """
    try:
        current_url = agent.page.url
        page_title = await agent.page.title()
        action_summary = generate_comprehensive_action_summary(agent.taken_actions, agent.predictions)
        
        termination_prompt = build_termination_prompt(agent.tasks[-1], page_title, current_url, len(agent.taken_actions), action_summary)
        
        # Check if screenshot exists before using it
        screenshot_for_termination = agent.screenshot_path if os.path.exists(agent.screenshot_path) else None
        response = await agent.engine.generate(prompt=[termination_prompt, termination_prompt, ""], image_path=screenshot_for_termination, turn_number=0)
        
        decision_text = ""
        if response is not None and (isinstance(response, list) and len(response) > 0) or (isinstance(response, str) and response):
            if isinstance(response, list):
                decision_text = (response[0] if response[0] is not None else "").strip()
            else:
                decision_text = (response or "").strip()
        
        if decision_text and decision_text.upper().startswith("TERMINATE:"):
            justification = decision_text[len("TERMINATE:"):].strip()
            agent.logger.info(f"LLM recommends task termination based on semantic analysis. Justification: {justification}")
            agent.complete_flag = True
            return True
        elif decision_text and decision_text.upper() == "CONTINUE":
            agent.logger.info("LLM recommends continuing based on semantic potential")
            return False
        
        # Default to continue if unclear response
        return False
        
    except Exception as e:
         agent.logger.error(f"LLM semantic termination analysis failed: {e}")
         return False

async def should_terminate_on_failure(agent, failure_type: str, error_message: str) -> bool:
    """
    Enhanced failure analysis using LLM semantic understanding instead of keyword matching.
    """
    try:
        # Get current page context for semantic analysis
        try:
            current_url = agent.page.url
            page_title = await agent.page.title()
            page_content = await agent.page.content()
            
            # Extract key indicators from page content
            page_indicators = {
                'url': current_url,
                'title': page_title,
                'has_error_message': any(term in page_content.lower() for term in ['error', 'forbidden', 'denied']),
                'has_login_prompt': any(term in page_content.lower() for term in ['sign in', 'login', 'authenticate'])
            }
        except Exception as page_error:
            agent.logger.warning(f"Could not analyze page context: {page_error}")
            page_indicators = {'url': 'unknown', 'title': 'unknown'}
        
        # Count recent failures for pattern detection
        recent_failures = sum(
            1 for action in agent.action_history[-8:] 
            if action.get('failure_type') == failure_type
        )
        
        # Use LLM to analyze if this represents a blocking condition
        blocking_analysis_prompt = build_blocking_analysis_prompt(failure_type, error_message, recent_failures, len(agent.taken_actions), page_indicators)
        
        # Use existing LLM infrastructure to analyze the failure
        try:
            analysis_response = await agent.engine.generate(
                prompt=["", blocking_analysis_prompt, ""],
                max_tokens=100,
                temperature=1.0,
                turn_number=0
            )
            
            if analysis_response and len(analysis_response) > 0:
                decision = analysis_response[0].upper() if isinstance(analysis_response, list) else str(analysis_response).upper()
                
                if "TERMINATE" in decision:
                    agent.logger.info(f"LLM recommends termination due to {failure_type} failure pattern or blocking condition")
                    return True
                elif "CONTINUE" in decision:
                    agent.logger.info(f"LLM recommends continuing despite {failure_type} failures")
                    return False
        except Exception as llm_error:
            agent.logger.warning(f"LLM failure analysis failed: {llm_error}")
        
        # Fallback: Conservative approach based on failure count and action count
        failure_threshold = 6
        action_threshold = 20
        
        if recent_failures >= failure_threshold and len(agent.taken_actions) >= action_threshold:
            agent.logger.info(f"High failure count ({recent_failures}) with sufficient actions ({len(agent.taken_actions)}) - recommending termination")
            return True
        
        # For fewer failures or early in execution, continue
        if len(agent.taken_actions) < 15:
            return False
            
        # Default to continue unless overwhelming evidence for termination
        return False
        
    except Exception as e:
        agent.logger.error(f"Failure analysis error: {e}")
        return False

async def verify_task_completion_before_terminate(agent) -> bool:
    """
    Enhanced task completion verification with comprehensive requirement analysis.
    Uses advanced LLM reasoning to understand task semantics and completion criteria.
    """
    try:
        current_url = agent.page.url
        page_title = await agent.page.title()
        
        # Generate complete action journey instead of compressed summary
        action_journey = generate_action_journey_summary(agent.taken_actions)
        
        # Capture current page state for analysis
        await agent.take_screenshot()
        
        # Build verification prompt
        checklist_context = ""
        if hasattr(agent, 'checklist_manager') and agent.checklist_manager.task_checklist:
            checklist_status = agent.checklist_manager.get_checklist_status_summary()
            checklist_context = f"\n**CHECKLIST STATUS**\n{checklist_status}\n"
        
        completion_verification_prompt = build_completion_verification_prompt(agent.tasks[-1], action_journey, page_title, current_url, checklist_context)
        
        # Provide last 3 screenshots for verification (if available)
        image_paths = []
        try:
            screenshots_dir = os.path.join(agent.main_path, 'screenshots')
            if os.path.isdir(screenshots_dir):
                # Collect by recent time steps
                for t in [agent.time_step, agent.time_step - 1, agent.time_step - 2]:
                    if t is not None and t >= 0:
                        p = os.path.join(screenshots_dir, f'screen_{t}.png')
                        if os.path.exists(p):
                            image_paths.append(p)
            # Fallback to current screenshot path if directory is missing
            if not image_paths and agent.screenshot_path and os.path.exists(agent.screenshot_path):
                image_paths = [agent.screenshot_path]
        except Exception:
            if agent.screenshot_path and os.path.exists(agent.screenshot_path):
                image_paths = [agent.screenshot_path]

        response = await agent.engine.generate(
            prompt=[completion_verification_prompt, completion_verification_prompt, ""], 
            image_paths=image_paths if image_paths else None, 
            turn_number=0
        )
        
        if response is not None and (isinstance(response, list) and len(response) > 0) or (isinstance(response, str) and response):
            # Safely handle response that might be None
            if isinstance(response, list):
                decision_text = response[0] if response[0] is not None else ""
            else:
                decision_text = str(response) if response is not None else ""
            
            decision = decision_text.upper().strip() if decision_text else ""
            
            # Check NOT_COMPLETED first to avoid substring matching bug
            if "NOT_COMPLETED" in decision or "NOT COMPLETED" in decision:
                agent.logger.info(f"Task completion rejected by LLM. Reason: {decision_text}. Continuing execution.")
                return False
            elif "COMPLETED" in decision:
                agent.logger.info(f"Task completion verified by LLM. Reason: {decision_text}. Terminating.")
                return True
        
        # If unclear response, use conservative approach
        agent.logger.warning("Unclear LLM response for task completion verification, defaulting to not completed")
        return False
        
    except Exception as e:
        agent.logger.error(f"Task completion verification failed: {e}")
        # Conservative approach: if verification fails, assume not completed
        return False

async def evaluate_task_success(agent) -> str:
    """
    Enhanced task completion evaluation - checks both executed actions and generated actions for completion signals.
    """
    if not agent.taken_actions:
        return "failure", "No actions taken."
    
    # Check if the last action was an explicit completion signal
    last_action = agent.taken_actions[-1] if agent.taken_actions else None
    if not last_action:
        return "failure", "Task in progress - no completion signal detected."
    
    # Check both executed action and generated action for completion signals
    last_action_type = last_action.get('action', '').upper()
    action_generation = str(last_action.get('action_generation', '')).upper()
    
    # Check for explicit completion signals in both executed and generated actions
    explicit_completion_signals = ['TERMINATE', 'COMPLETE', 'DONE', 'FINISH']
    is_explicit_completion = (
        any(signal in last_action_type for signal in explicit_completion_signals) or
        any(signal in action_generation for signal in explicit_completion_signals)
    )
    
    # Also check for completion phrases in action generation, reasoning, and action text
    completion_phrases = [
        'task completed', 'task finished', 'task done', 'successfully completed',
        'objective achieved', 'goal accomplished', 'requirements met', 'task fulfilled',
        'completed successfully', 'finished successfully', 'done successfully',
        'task completed successfully', 'task finished successfully', 'task done successfully',
        'successfully finished', 'successfully done', 'task is completed', 'task is finished',
        'task is done', 'task has been completed', 'task has been finished', 'task has been done'
    ]
    
    action_text = str(last_action.get('action_generation', '')).lower()
    reasoning_text = str(last_action.get('reasoning', '')).lower()
    action_value = str(last_action.get('value', '')).lower()
    combined_text = f"{action_text} {reasoning_text} {action_value}"
    
    has_completion_phrase = any(phrase in combined_text for phrase in completion_phrases)
    
    # Enhanced completion detection: also check recent actions for completion signals
    recent_completion_detected = False
    if len(agent.taken_actions) >= 2:
        for recent_action in agent.taken_actions[-3:]:  # Check last 3 actions
            recent_generation = str(recent_action.get('action_generation', '')).lower()
            if any(phrase in recent_generation for phrase in completion_phrases):
                recent_completion_detected = True
                agent.logger.info(f"Completion signal detected in recent action: {recent_generation}")
                break
    
    # If no completion signal detected, use LLM to generate natural description
    if not is_explicit_completion and not has_completion_phrase and not recent_completion_detected:
        agent.logger.info("No explicit completion signal detected - generating LLM description")
        
        try:
            current_url = agent.page.url if agent.page else "Unknown URL"
            page_title = agent.page.title() if agent.page else "Unknown Page"
            
            # Generate comprehensive action summary
            action_summary = generate_comprehensive_action_summary(agent.taken_actions, agent.predictions)
            
            # Create prompt for LLM to generate natural description
            if not action_summary:
                action_summary = "No actions taken."

            description_prompt = build_description_prompt(agent.tasks[-1] if agent.tasks else "Unknown task", page_title, current_url, len(agent.taken_actions), action_summary)
            
            # Use LLM to generate natural description
            screenshot_for_desc = agent.screenshot_path if os.path.exists(agent.screenshot_path) else None
            description_response = await agent.engine.generate(
                prompt=[description_prompt, description_prompt, ""], 
                image_path=screenshot_for_desc, 
                turn_number=0
            )
            
            if description_response and len(description_response) > 0:
                natural_description = (description_response[0] if isinstance(description_response, list) else description_response or "").strip()
                if natural_description:
                    agent.logger.info(f"Generated natural description: {natural_description}")
                    return "failure", natural_description
            
            # Fallback if LLM generation fails
            return "failure", f"The agent has taken {len(agent.taken_actions)} actions while working on the task. Currently on {page_title} page, but the task has not been completed yet."
            
        except Exception as e:
            agent.logger.error(f"Error generating LLM description: {str(e)}")
            return "failure", f"The agent is working on the task and has taken {len(agent.taken_actions)} actions so far. The task is still in progress."
    
    # Only now perform comprehensive LLM evaluation since Agent signaled completion
    try:
        # Get comprehensive page context
        current_url = agent.page.url
        page_title = agent.page.title
        
        # Extract comprehensive action summary with all historical information
        action_summary = generate_comprehensive_action_summary(agent.taken_actions, agent.predictions)
        
        # Include additional context for evaluation
        additional_context = []
        
        # Add session information with enhanced safety checks (HTML responses only)
        if (hasattr(agent, 'session_control') and agent.session_control and 
            isinstance(agent.session_control, dict) and 
            agent.session_control.get('last_response')):
            try:
                lr = agent.session_control['last_response']
                ct = str(lr.get('headers', {}).get('content-type', '')).lower()
                if 'text/html' in ct:
                    additional_context.append(f"Last HTTP Response: {lr}")
            except Exception:
                pass
        
        # Add continuous no-op count
        if hasattr(agent, 'continuous_no_op'):
            additional_context.append(f"Continuous No-Op Count: {agent.continuous_no_op}")
        
        # Add complete flag status
        additional_context.append(f"Complete Flag Status: {getattr(agent, 'complete_flag', False)}")
        
        context_info = "\n".join(additional_context) if additional_context else "No additional context available."
        
        # Generate action journey summary for evaluation
        action_journey = generate_action_journey_summary(agent.taken_actions, agent.predictions)
        
        # Updated evaluation prompt to be more rigorous
        eval_prompt = build_evaluation_prompt(agent.tasks[-1], action_journey, page_title, current_url, len(agent.taken_actions))
        
        # Use LLM to evaluate task completion with enhanced prompt including full history
        # Check if screenshot exists before using it
        screenshot_for_eval = agent.screenshot_path if os.path.exists(agent.screenshot_path) else None
        evaluation = await agent.engine.generate(prompt=[eval_prompt, eval_prompt, ""], image_path=screenshot_for_eval, turn_number=0)
        
        # Extract and validate the evaluation result
        if evaluation and len(evaluation) > 0:
            evaluation_text = (evaluation[0] if isinstance(evaluation, list) else evaluation or "").strip()
            
            # Extract status from the response (supports English only)
            status_line_pattern = r"(STATUS)\s*[:：]\s*(success|failure|partial)"
            m = re.search(status_line_pattern, evaluation_text, flags=re.IGNORECASE)
            raw_status = (m.group(2).lower() if m else None)
            if raw_status in {"success", "failure", "partial"}:
                status = raw_status
            else:
                # Heuristic fallback: detect success keywords if explicit status line missing
                lowered = evaluation_text.lower()
                if "success" in lowered and "fail" not in lowered:
                    status = "success"
                else:
                    status = "failure"
            
            # Extract the natural description (everything before STATUS line)
            if "STATUS:" in evaluation_text.upper():
                description = evaluation_text.split("STATUS:")[0].strip()
            else:
                description = evaluation_text
            
            # Clean up the description
            if not description:
                description = "Agent signaled completion but evaluation description was not generated properly."
            
            agent.logger.info(f"Task evaluation after Agent completion signal: {status} - {description}")
            return status, description
        else:
            # If LLM evaluation fails but Agent signaled completion, give benefit of doubt
            agent.logger.warning("LLM evaluation failed but Agent signaled completion")
            return "failure", "The agent signaled that it completed the task, but the evaluation process encountered an issue and couldn't provide a detailed assessment."
            
    except Exception as e:
        agent.logger.error(f"LLM task evaluation failed: {e}")
        # If evaluation fails but Agent signaled completion, don't fail the task
        return "failure", f"The agent indicated it completed the task, but there was an issue during the evaluation process: {str(e)}"

async def review_action_generation(agent, action_generation_output):
    """
    Enhanced action generation output review and loop detection using LLM-based intelligent analysis.
    """
    try:
        if not action_generation_output:
            agent.logger.warning("Empty action generation output")
            return False
        
        # Basic output validation
        output_lower = action_generation_output.lower()
        if len(output_lower) < 3:
            agent.logger.warning("Action generation output too short")
            return False
        
        # Check for recognizable action keywords
        action_keywords = ['click', 'type', 'scroll', 'hover', 'select', 'drag', 'press', 'wait', 'go', 'navigate']
        has_action_keyword = any(keyword in output_lower for keyword in action_keywords)
        
        if not has_action_keyword:
            agent.logger.warning("No recognizable action keywords found in output")
            # Allow PRESS ENTER as a special case
            if 'press' in output_lower or 'enter' in output_lower:
                agent.logger.info("Detected PRESS ENTER action, allowing execution")
                return True
            return False
        
        agent.logger.info("Action generation review passed")
        return True
        
    except Exception as e:
        agent.logger.error(f"Action generation review error: {e}")
        return True  # Default to continue on review error
