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

import asyncio
import json
import logging
import os
import random
import time
import traceback
from datetime import datetime
import sys
import re

import toml
from playwright.async_api import Locator
from seeact.agent.config import load_agent_config
from seeact.agent.reporting import (
    generate_comprehensive_action_summary, 
    generate_recent_action_summary, 
    generate_action_description, 
    compose_action_description, 
    save_results, 
    emergency_save
)
from seeact.agent.evaluation import (
    should_terminate_intelligently, 
    should_terminate_on_failure, 
    verify_task_completion_before_terminate, 
    evaluate_task_success, 
    review_action_generation
)

from seeact.prompts.utils import get_index_from_option_name, generate_new_query_prompt, \
    generate_new_referring_prompt, format_options, generate_option_name, analyze_repetitive_patterns, \
    generate_action_journey_summary, llm_summarize_actions, llm_update_history_summary, \
    initialize_prompts
from seeact.browser.helper import saveconfig, setup_agent_logger, page_on_close_handler, page_on_response_handler, \
    page_on_open_handler, page_on_crash_handler, save_action_history, \
    start_agent_browser, stop_agent_browser, start_playwright_tracing, stop_playwright_tracing, \
    save_traces, get_page, set_page
from seeact.prompts.format import format_choices, postprocess_action_lmm, postprocess_action_lmm_pixel
from seeact.utils.image import take_screenshot, annotate_current_screenshot, take_full_page_screenshot_with_cropping
from seeact.llm.engine import engine_factory
from seeact.llm.engine import LLM_IO_RECORDS, add_llm_io_record
from seeact.managers.checklist import ChecklistManager
from seeact.utils.reasoning import generate_task_reasoning, format_reasoning_for_prompt
from seeact.browser.dom import (
    extract_typeable_elements,
    extract_selectable_elements,
    choose_field_with_llm
)
from seeact.browser.recovery import capture_page_state, detect_page_state_change, find_click_target_by_text, \
    analyze_previous_action_results, add_action_to_stack, detect_repetitive_actions, \
    manage_action_history, is_action_forbidden, \
    is_page_blocked_or_blank
from .executor import perform_action, execute
from .predictor import predict


class SeeActAgent:
    def __init__(self,
                 config_path=None,
                 config=None,  # Add config parameter
                 save_file_dir="seeact_agent_files",
                 default_task='Find the pdf of the paper "GPT-4V(ision) is a Generalist Web Agent, if Grounded"',
                 default_website="https://www.google.com/",
                 input_info=["screenshot"],
                 crawler_mode=False,
                 crawler_max_steps=20,  # Increased from 10 to 20 to allow more exploration
                 max_auto_op=30,
                 max_continuous_no_op=15,
                 highlight=False,
                 headless=False,
                 args=[],
                 browser_app="chrome",
                 persistant=False,
                 persistant_user_path="",
                 save_video=False,
                 viewport={
                     "width": 1280,
                     "height": 720
                 },
                 stealth_mode=True,
                 user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0",
                 tracing=False,
                 trace={
                     "screenshots": True,
                     "snapshots": True,
                     "sources": True
                 },
                 rate_limit=-1,
                 model="openrouter/qwen/qwen-2.5-72b-instruct",
                 temperature=1.0,
                 create_timestamp_dir=True,
                 task_id=None  # Add task_id parameter
                 ):

        self.config = load_agent_config(
            config_path=config_path,
            config=config,
            save_file_dir=save_file_dir,
            default_task=default_task,
            default_website=default_website,
            input_info=input_info,
            crawler_mode=crawler_mode,
            crawler_max_steps=crawler_max_steps,
            max_auto_op=max_auto_op,
            max_continuous_no_op=max_continuous_no_op,
            highlight=highlight,
            headless=headless,
            args=args,
            browser_app=browser_app,
            persistant=persistant,
            persistant_user_path=persistant_user_path,
            save_video=save_video,
            viewport=viewport,
            stealth_mode=stealth_mode,
            user_agent=user_agent,
            tracing=tracing,
            trace=trace,
            model=model,
            temperature=temperature
        )

        self.complete_flag = False
        self.session_control = {
            'active_page': None,
            'context': None,
            'browser': None
        }
        self.default_task = default_task
        self.tasks = [self.default_task]
        
        # Use provided task_id - should always be provided by caller
        if task_id is None:
            raise ValueError("task_id must be provided and cannot be None")
        self.task_id = task_id

        # Create directory structure similar to old implementation
        if create_timestamp_dir:
            base_dir = os.path.join(self.config["basic"]["save_file_dir"], datetime.now().strftime("%Y%m%d_%H%M%S"))
        else:
            base_dir = self.config["basic"]["save_file_dir"]
        
        # Create task_id subdirectory like in old implementation
        self.main_path = os.path.join(base_dir, self.task_id)
        os.makedirs(self.main_path, exist_ok=True)
        self.action_space = ["CLICK", "KEYBOARD", "PRESS ENTER", "WAIT", "HOVER", "SCROLL UP", "SCROLL DOWN", "SCROLL TOP", "SCROLL BOTTOM", "NEW TAB", "CLOSE TAB",
                             "GO BACK", "GO FORWARD",
                             "TERMINATE", "SELECT", "TYPE", "GOTO", "NONE"]  # Define the list of actions here

        self.no_value_op = ["CLICK", "PRESS ENTER", "WAIT", "HOVER", "SCROLL UP", "SCROLL DOWN", "SCROLL TOP", "SCROLL BOTTOM", "NEW TAB", "CLOSE TAB",
                            "PRESS HOME", "PRESS END", "PRESS PAGEUP", "PRESS PAGEDOWN",
                            "GO BACK", "GO FORWARD", "TERMINATE", "NONE"]

        self.with_value_op = ["SELECT", "TYPE", "KEYBOARD", "GOTO", "SAY"]
        self.last_click_coordinates = None
        self.last_click_viewport_coords = None
        self.initial_frame_saved = False
        self._current_coordinates_type = 'normalized'

        self.no_element_op = ["PRESS ENTER", "WAIT", "KEYBOARD", "SCROLL UP", "SCROLL DOWN", "SCROLL TOP", "SCROLL BOTTOM", "NEW TAB", "CLOSE TAB", "GO BACK", "GOTO",
                              "PRESS HOME", "PRESS END", "PRESS PAGEUP", "PRESS PAGEDOWN",
                              "GO FORWARD",
                              "TERMINATE", "NONE", "SAY"]

        # Initialize the primary logger and the developer logger with error handling
        try:
            self.logger = self._setup_logger(redirect_to_dev_log=False)
        except Exception as e:
            # Create a fallback logger that only logs to console if file logging fails
            print(f"Warning: Failed to create file logger: {e}")
            print("Creating fallback console-only logger...")
            self.logger = logging.getLogger(f"{self.task_id}_fallback")
            self.logger.setLevel(logging.INFO)
            self.logger.handlers.clear()
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(asctime)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            self.logger.propagate = False
            self.logger.warning(f"Using fallback console-only logger due to file logger creation failure: {e}")
        
        # self.dev_logger = self._setup_dev_logger()

        # # Redirect primary logger messages to dev_logger as well
        # for handler in self.logger.handlers:
        #     self.dev_logger.addHandler(handler)

        # Initialize engine with error handling
        try:
            # Get engine configuration - prioritize new structure over legacy
            model_config = self.config.get('model', {})
            api_keys_config = self.config.get('api_keys', {})
            
            # Prepare engine parameters
            engine_params = {}
            
            # Model name
            engine_params['model'] = model_config.get('name', 'openrouter/qwen/qwen-2.5-72b-instruct')
            
            # Temperature
            engine_params['temperature'] = model_config.get('temperature', 1)
            
            # API key - OpenRouter/Gemini only
            api_key = None
            model_name = engine_params['model'].lower()
            
            # Prefer config-provided keys over environment to honor per-mode settings
            if 'claude' in model_name:
                api_key = api_keys_config.get('openrouter_api_key') or os.getenv('OPENROUTER_API_KEY')
            elif 'gemini' in model_name:
                api_key = (
                    api_keys_config.get('openrouter_api_key') or os.getenv('OPENROUTER_API_KEY') or 
                    api_keys_config.get('gemini_api_key') or os.getenv('GEMINI_API_KEY')
                )
            else:
                api_key = api_keys_config.get('openrouter_api_key') or os.getenv('OPENROUTER_API_KEY')
            
            if api_key:
                engine_params['api_key'] = api_key
            
            self.engine = engine_factory(**engine_params)
            try:
                setattr(self.engine, 'task_id', self.task_id)
            except Exception:
                pass
            
            # Store model and temperature as instance attributes for predict() method
            self.model = engine_params.get('model', 'openrouter/qwen/qwen-2.5-72b-instruct')
            self.temperature = engine_params.get('temperature', 1.0)
            
            self.logger.info("Engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize engine: {e}")
            self.logger.warning("Agent will continue with limited functionality")
            self.engine = None
            self.model = 'openrouter/qwen/qwen-2.5-72b-instruct'
            self.temperature = 1.0
        
        # Initialize checklist engine and ChecklistManager
        try:
            checklist_model = model_config.get('checklist_model', 'openrouter/qwen/qwen3-vl-8b-instruct')
            checklist_engine_params = {'model': checklist_model, 'temperature': 0.7}
            if api_key:
                checklist_engine_params['api_key'] = api_key
            checklist_engine = engine_factory(**checklist_engine_params)
            self.logger.info(f"Checklist engine initialized with model: {checklist_model}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize checklist engine: {e}, using main engine")
            checklist_engine = self.engine
        
        self.checklist_manager = ChecklistManager(
            engine=self.engine,
            checklist_engine=checklist_engine,
            logger=self.logger
        )
        
        # Remove dedicated GUI grounding model; main model handles grounding.
        self.taken_actions = []
        self.action_history = []  # Track action history for failure analysis
        self.action_summaries = []  # Store natural language summaries every 5 steps
        
        # Action stack for preventing repetitive actions
        self.action_stack = []
        
        # Task reasoning from reasoning model
        self.task_reasoning = ""  # Strategic guidance generated at task start  # Stack to track recent actions for repetition detection
        self.max_stack_size = 5  # Keep track of last 5 actions
        self.forbidden_actions = set()  # Set of forbidden action patterns
        
        # Loop detection for preventing repetitive actions
        self.query_generation_count = 0  # Track query generation attempts
        self.max_query_generations = 5  # Maximum query generation attempts
        # Note: Removed repeated_action_threshold - now using LLM-based judgment
        
        # Initialize dynamic checklist system (managed by ChecklistManager)
        self.checklist_generated = False  # Flag to track if checklist has been generated

        # Action history management settings
        self.max_action_history = 50  # Maximum number of actions to keep

        # Initialize unified prompts for mixed architecture
        self.prompts = self._initialize_prompts()
        self.time_step = 0
        self.valid_op = 0
        self.continuous_no_op = 0
        self.predictions = []
        self.visited_links = []
        self._page = None
        
        # Initialize screenshot path - will be updated when screenshots are taken
        # Initialize screenshot path tracking
        self._screenshot_path = None

        self.history_summary_interval = 5
        self.history_recent_window = self.config.get('agent', {}).get('history_recent_window', 5)
        self.llm_history_summary_text = ""
        self.llm_summary_covered_steps = 0
        self.is_stopping = False


    async def _is_page_blocked_or_blank(self):
        return await is_page_blocked_or_blank(self.page, logger=self.logger)

    async def _generate_task_reasoning(self):
        """
        Generate strategic reasoning about the task using a reasoning model.
        Called at the start of task execution before browser launches.
        """
        model_config = self.config.get('model', {})
        reasoning_model = model_config.get('reasoning_model', self.model)
        reasoning_temp = model_config.get('reasoning_temperature', 1.0)
        enable_thinking = model_config.get('reasoning_enable_thinking_mode', True)
        enable_online = model_config.get('reasoning_enable_online', True)
        reasoning_effort = model_config.get('reasoning_effort', 'high')
        reasoning_verbosity = model_config.get('reasoning_verbosity', 'high')
        reasoning_web_search = model_config.get('reasoning_enable_web_search', False)
        
        # Get current task and website
        current_task = self.tasks[-1] if self.tasks else self.default_task
        # Use actual_website if available (set in start()), otherwise use config default
        current_website = getattr(self, 'actual_website', None) or self.config.get("basic", {}).get("default_website")
        
        self.logger.info("="*70)
        self.logger.info("🧠 TASK REASONING PHASE")
        self.logger.info("="*70)
        self.logger.info(f"Task: {current_task}")
        self.logger.info(f"Website: {current_website}")
        self.logger.info(f"Reasoning Model: {reasoning_model}")
        
        # Build soft constraints for reasoning
        try:
            from urllib.parse import urlparse
            allowed_domain = urlparse(current_website).hostname or ""
        except Exception:
            allowed_domain = ""
        from seeact.prompts.templates import build_task_constraints_prompt
        constraints_text = build_task_constraints_prompt(
            allowed_domain=allowed_domain,
            disallow_login=True,
            disallow_offsite=True,
            extra_rules=""
        )
        plugins_payload = None
        result = await generate_task_reasoning(
            task_description=current_task,
            website=current_website,
            model=reasoning_model,
            enable_thinking=enable_thinking,
            enable_online=enable_online,
            reasoning_effort=reasoning_effort,
            reasoning_verbosity=reasoning_verbosity,
            use_web_search=reasoning_web_search,
            temperature=reasoning_temp,
            logger=self.logger,
            policy_constraints=constraints_text,
            plugins=plugins_payload,
            task_id=self.task_id
        )
        
        if result['success']:
            self.task_reasoning = result['reasoning']
            self.logger.info("="*70)
            self.logger.info("✅ Reasoning generated successfully")
            self.logger.info("="*70)
        else:
            self.logger.warning(f"⚠️ Failed to generate reasoning: {result['error']}")
            self.task_reasoning = ""  # Continue without reasoning
        
    async def generate_task_checklist(self, task_description):
        """Delegate to ChecklistManager."""
        result = await self.checklist_manager.generate_task_checklist(task_description, self.task_reasoning)
        self.checklist_generated = self.checklist_manager.checklist_generated
        return result

    # Delegate to ChecklistManager - use property for backward compatibility
    @property
    def task_checklist(self):
        """Access checklist through manager for backward compatibility."""
        return self.checklist_manager.task_checklist

    def update_checklist_item(self, item_id, status, description=None):
        """Delegate to ChecklistManager."""
        return self.checklist_manager.update_checklist_item(item_id, status, description)

    def get_checklist_status(self):
        return self.checklist_manager.get_checklist_status()

    def format_checklist_for_prompt(self):
        """Delegate to ChecklistManager."""
        return self.checklist_manager.format_checklist_for_prompt()

    async def _update_checklist_after_action(self, action_data):
        """Delegate to ChecklistManager."""
        if not self.task_checklist:
            return
            
        # Get current page context
        current_url = self.page.url if hasattr(self, 'page') and self.page else "Unknown"
        page_title = ""
        try:
            if hasattr(self, 'page') and self.page:
                page_title = await self.page.title()
        except:
            page_title = "Unknown"
        
        # Get page state
        page_state = {}
        try:
            if hasattr(self, 'page') and self.page:
                page_state = await self._capture_page_state()
        except Exception:
            pass
        
        # Collect latest two screenshots for multimodal checklist update
        image_paths = []
        try:
            if self.screenshot_path and os.path.exists(self.screenshot_path):
                image_paths.append(self.screenshot_path)
            prev_path = os.path.join(self.main_path, 'screenshots', f'screen_{max(0, self.time_step-1)}.png')
            if os.path.exists(prev_path):
                image_paths.append(prev_path)
        except Exception:
            pass

        # Delegate to checklist manager
        await self.checklist_manager.update_checklist_after_action(
            action_data,
            current_url,
            page_title,
            page_state,
            self.action_history,
            image_paths=image_paths
        )

    def add_checklist_item(self, description, item_id=None):
        """Delegate to ChecklistManager."""
        return self.checklist_manager.add_checklist_item(description, item_id)

    def remove_checklist_item(self, item_id):
        """Delegate to ChecklistManager."""
        return self.checklist_manager.remove_checklist_item(item_id)
        

    def _initialize_prompts(self):
        return initialize_prompts()

    def update_action_space(self, new_actions):
        """Update the action space and regenerate the action_format prompt."""
        if isinstance(new_actions, list) and all(isinstance(item, str) for item in new_actions):
            self.action_space = new_actions
            self.prompts["action_format"] = f"ACTION: Choose an action from {{{', '.join(self.action_space)}}}."
        else:
            print("Invalid action space provided. It must be a list of strings.")



    def _setup_logger(self, redirect_to_dev_log=False):
        return setup_agent_logger(self.task_id, self.main_path, redirect_to_dev_log=redirect_to_dev_log)

    async def page_on_close_handler(self):
        return await page_on_close_handler(self)

    def save_action_history(self, filename="action_history.txt"):
        """Save the history of taken actions to a file in the main path."""
        return save_action_history(self.main_path, self.taken_actions, self.logger, filename=filename)

    async def page_on_navigation_handler(self, frame):
        # Simplified navigation handler similar to seeact_old2.py
        return await page_on_navigation_handler(frame)
        

    async def page_on_crash_handler(self, page):
        return await page_on_crash_handler(self, page)

    async def page_on_response_handler(self, response):
        """Handle HTTP responses and store the latest response info"""
        return await page_on_response_handler(self, response)

    async def page_on_open_handler(self, page):
        return await page_on_open_handler(self, page)

    async def start(self, headless=None, args=None, website=None):
        return await start_agent_browser(self, headless=headless, args=args, website=website)

    def update_prompt_part(self, part_name, new_text):
        """Update the specified part of the prompt information."""
        if part_name in self.prompts:
            self.prompts[part_name] = new_text
            return True
        else:
            print(f"Prompt part '{part_name}' not found.")
            return False

    def generate_prompt(self, task=None, previous=None, choices=None, reflection_analysis=None):
        """Generate a unified prompt for hybrid grounding architecture."""
        prompt_list = []
        
        # Detect repetitive actions before generating prompt
        repetition_detection = self._detect_repetitive_actions()
        
        # Hybrid approach: always include both visual and element-based capabilities
        system_prompt_input = self.prompts["system_prompt"]
        action_space_input = self.prompts["action_space"]
        question_description_input = self.prompts["question_description"]
        
        previous_ = self.taken_actions if self.taken_actions else None
        
        # Include checklist in the system prompt if available
        checklist_context = ""
        if self.task_checklist:
            checklist_context = "\n\n" + self.format_checklist_for_prompt() + "\n"
            self.logger.info("Including checklist in prompt generation")
        
        # Include reflection analysis if available
        reflection_context = ""
        if reflection_analysis:
            reflection_context = f"\n\n**REFLECTION ANALYSIS FROM PREVIOUS ACTIONS**:\n{reflection_analysis}\n"
            self.logger.info("Including reflection analysis in prompt generation")
        
        # Unified prompt generation regardless of grounding strategy
        prompt_list.extend(
            generate_new_query_prompt(
                system_prompt=system_prompt_input + action_space_input + checklist_context + reflection_context,
                task=self.tasks[-1], 
                previous_actions=previous_,
                                      question_description=question_description_input,
                repetition_detection=repetition_detection
            )
        )
        
        # Only add referring prompt if choices are provided (element-based approach)
        if choices is not None:
            referring_input = self.prompts["referring_description"]
            element_format_input = self.prompts["element_format"]
            action_format_input = self.prompts["action_format"]
            value_format_input = self.prompts["value_format"]
            prompt_list.append(
                generate_new_referring_prompt(referring_description=referring_input,
                                            element_format=element_format_input,
                                            action_format=action_format_input, value_format=value_format_input,
                                            choices=choices))

        return prompt_list

    # Removed _decide_grounding_strategy function - using fixed strategy from config instead

    async def perform_action(self, target_element=None, action_name=None, value=None, target_coordinates=None,
                             element_repr=None, field_name=None, action_description=None, clear_first: bool = True,
                             press_enter_after: bool = False):
        return await perform_action(
            self,
            target_element=target_element,
            action_name=action_name,
            value=value,
            target_coordinates=target_coordinates,
            element_repr=element_repr,
            field_name=field_name,
            action_description=action_description,
            clear_first=clear_first,
            press_enter_after=press_enter_after
        )

    async def predict(self):
        """
        Generate a prediction for the next action using a unified tool-calling format.
        Single LLM call returns both reasoning and action in one response.
        Always returns a valid prediction dictionary, never None.
        """
        return await predict(self)

    def _generate_action_description(self, parsed_action):
        """
        Generate a human-readable description from parsed action.
        
        Args:
            parsed_action: Dict with 'action', optional 'coordinates', 'text', 'value', 'field'
            
        Returns:
            str: Human-readable action description
        """
        return generate_action_description(parsed_action, self.logger)

    def _compose_action_description(self, action, value, field, element_desc, coords=None):
        return compose_action_description(action, value, field, element_desc, coords)

# removed: handle_grounding_failure
                
    

    

    
    


    
    # removed: get_expanded_elements
    

    


    # removed: get_element_data_relaxed
    
    # removed: predict_with_expanded_elements
    
    
    


    
    # removed: extract_target_from_action_generation


    

    

    

    


    async def execute(self, prediction_dict):
        """
        Execute the predicted action on the webpage.
        """
        return await execute(self, prediction_dict)

    async def stop(self):
        await stop_agent_browser(self)

        # Prepare data for saving - use safe defaults if anything fails
        action_history_for_output = []
        success_status = "error"
        final_result_response = "Unknown error occurred during task execution"
        
        try:
            # Convert enhanced actions to a more readable format for final output
            for action in self.taken_actions:
                if isinstance(action, dict):
                    combined_desc = action.get('action_description', '')
                    elem_desc = action.get('element_description', '')
                    if elem_desc:
                        combined_desc = f"{combined_desc} | Element: {elem_desc}"
                    action_history_for_output.append({
                        "step": action.get('step', 'N/A'),
                        "action_generation": action.get('action_generation_response', ''),
                        "action_grounding": action.get('action_grounding_response', ''),
                        "action": action.get('predicted_action', ''),
                        "value": action.get('predicted_value', ''),
                        "element": action.get('element_description', ''),
                        "error": action.get('error', ''),
                        "description": combined_desc,
                        "coordinates": action.get('coordinates'),
                        "element_center": action.get('element_center'),
                        "element_box": action.get('element_box')
                    })
                else:
                    action_history_for_output.append(str(action))
        except Exception as e:
            self.logger.error(f"Error processing action history: {e}")
            # Use minimal fallback data
            action_history_for_output = [f"Error processing action history: {str(e)}"]

        try:
            # Task success evaluation is disabled
            success_status = "unknown"
            final_result_response = "Evaluation disabled"
        except Exception as e:
            self.logger.error(f"Error evaluating task success: {e}")
            success_status = "error"
            final_result_response = "Unable to determine task completion status due to evaluation error. Please check the action history for task progress."

        # Create final JSON with safe defaults
        final_json = {
            "confirmed_task": self.default_task if hasattr(self, 'default_task') and self.default_task else "Unknown task",
            "website": getattr(self, 'actual_website', self.config.get("basic", {}).get("default_website", "Unknown website")) if hasattr(self, 'config') and self.config else "Unknown website",
            "task_id": getattr(self, 'task_id', 'demo_task'),
            "success_or_not": success_status,
            "final_result_response": final_result_response,
            "num_step": len(self.taken_actions) if hasattr(self, 'taken_actions') else 0,
            "action_history": action_history_for_output,
            "exit_by": "Task completed"
        }

        # Delegate saving to reporting module
        save_results(
            main_path=self.main_path,
            task_id=self.task_id,
            final_json=final_json,
            taken_actions=self.taken_actions,
            config=self.config,
            logger=self.logger,
            llm_io_records=LLM_IO_RECORDS
        )

    def _emergency_save(self, error_info="Unknown error"):
        """
        Emergency save mechanism when normal save operations fail.
        Saves minimal data to ensure no information is lost.
        """
        return emergency_save(self.task_id, self.taken_actions, error_info, self.logger)

    async def _evaluate_task_success(self) -> str:
        """
        Enhanced task completion evaluation - checks both executed actions and generated actions for completion signals.
        This ensures completion signals in action_generation are properly recognized even if action grounding fails.
        """
        return await evaluate_task_success(self)
    
    def _generate_recent_action_summary(self) -> str:
        """
        Generate a simplified summary of recent actions for evaluation.
        Only includes essential information: action type, target, and result.
        """
        return generate_recent_action_summary(self.taken_actions)

    def _generate_comprehensive_action_summary(self, max_actions=20, compress_old=True) -> str:
        """
        Generate a comprehensive but compressed summary of actions taken for evaluation context.
        Includes compressed historical information with intelligent truncation to reduce token usage.
        """
        return generate_comprehensive_action_summary(self.taken_actions, self.predictions, getattr(self, 'reflection_history', None), max_actions, compress_old)
    
    async def _should_terminate_intelligently(self) -> bool:
        """
        Ultra-conservative intelligent termination - heavily biased toward continuation.
        Only terminates when Agent explicitly signals completion or in extreme failure cases.
        """
        return await should_terminate_intelligently(self)
    
    async def _should_terminate_on_failure(self, failure_type: str, error_message: str) -> bool:
        """
        Enhanced failure analysis using LLM semantic understanding instead of keyword matching.
        Focuses on understanding the context and nature of failures rather than hardcoded patterns.
        """
        return await should_terminate_on_failure(self, failure_type, error_message)
    
    def _parse_say_content(self, say_text: str) -> dict:
        """
        Parse SAY content for Chain of Thought (CoT) analysis.
        
        NOTE: This function is for analyzing SAY content as thinking steps only.
        SAY actions no longer trigger task completion - they are purely for reasoning.
        """
        import json
        import re
        
        if not say_text:
            return {'type': 'empty'}
        
        say_lower = say_text.lower().strip()
        
        # Try to parse as JSON first
        try:
            if say_text.strip().startswith('{') and say_text.strip().endswith('}'):
                parsed_json = json.loads(say_text)
                return {
                    'type': 'json',
                    'content': parsed_json,
                    'original': say_text
                }
        except json.JSONDecodeError:
            pass
        
        # NOTE: Removed completion signal detection since SAY no longer triggers completion
        # SAY actions are now purely for Chain of Thought (thinking steps)
        
        # Check for instruction patterns (thinking about next steps)
        instruction_patterns = [
            r'please\s+(\w+)', r'next\s+step', r'should\s+(\w+)', 
            r'need\s+to\s+(\w+)', r'must\s+(\w+)', r'action:\s*(\w+)'
        ]
        
        for pattern in instruction_patterns:
            match = re.search(pattern, say_lower)
            if match:
                return {
                    'type': 'thinking_instruction',
                    'instruction': match.group(1) if match.groups() else say_text,
                    'original': say_text
                }
        
        # Check for analysis patterns (thinking about current state)
        analysis_indicators = [
            'analyzing', 'considering', 'thinking', 'evaluating', 'assessing',
            'reviewing', 'examining', 'checking', 'looking at', 'observing'
        ]
        
        if any(indicator in say_lower for indicator in analysis_indicators):
            return {
                'type': 'thinking_analysis',
                'message': say_text,
                'original': say_text
            }
        
        # Default: general thinking step
        return {
            'type': 'thinking_general',
            'message': say_text,
            'original': say_text
        }

    async def _verify_task_completion_before_terminate(self) -> bool:
        """
        Enhanced task completion verification with comprehensive requirement analysis.
        Uses advanced LLM reasoning to understand task semantics and completion criteria.
        """
        return await verify_task_completion_before_terminate(self)

    async def review_action_generation(self, action_generation_output):
        """
        Enhanced action generation output review and loop detection using LLM-based intelligent analysis.
        """
        return await review_action_generation(self, action_generation_output)
    
    # Removed _llm_analyze_action_repetition function as repetition detection 
    # is now integrated directly into the LLM action generation prompt
    


    def clear_action_history(self):
        """
        Clears the history of actions taken by the agent.
        """
        self.taken_actions.clear()
        self.action_history.clear()  # Also clear action_history to prevent memory leak
        self.logger.info("Cleared action history and action_history.")

    def reset_comlete_flag(self, flag=False):
        self.complete_flag = flag

    def change_task(self, new_task, clear_history=False):
        """
        Changes the task requirement for the agent.

        Parameters:
        - new_task: The new task requirement as a string.
        """
        if new_task and isinstance(new_task, str):

            self.logger.info(f"Changed task from {self.tasks[-1]} to: {new_task}")
            self.tasks.append(new_task)
            # Optionally clear action history when changing task
            if clear_history:
                self.clear_action_history()
            else:
                task_change_action = {
                    "step": self.time_step,
                    "action_description": f"Changed task from {self.tasks[-2]} to: {new_task}",
                    "action_generation_response": "",
                    "action_grounding_response": "",
                    "predicted_action": "TASK_CHANGE",
                    "predicted_value": new_task,
                    "element_description": "",
                    "success": True,
                    "error": None,
                    "http_response": {},
                    "page_content_summary": "Task changed successfully"
                }
                self.taken_actions.append(task_change_action)
                # Also add to action_history for failure analysis
                self.action_history.append(task_change_action)

        else:
            self.logger.info("Invalid new task. It must be a non-empty string.")

        # Optionally, you can save the taken_actions to a file or database for record-keeping

    # ADD no op count and op count, add limit to op

    # decompose run to predict and execute.

    async def take_screenshot(self):
        return await take_screenshot(self)

    async def _annotate_current_screenshot(self):
        return await annotate_current_screenshot(self)

    async def _take_full_page_screenshot_with_cropping(self, target_elements=None, screenshot_path=None):
        return await take_full_page_screenshot_with_cropping(self, target_elements=target_elements, screenshot_path=screenshot_path)

    async def start_playwright_tracing(self):
        return await start_playwright_tracing(self)

    async def stop_playwright_tracing(self):
        return await stop_playwright_tracing(self)

    async def save_traces(self):
        return await save_traces(self)

    @property
    def page(self):
        return get_page(self)

    @page.setter
    def page(self, value):
        set_page(self, value)

    @property
    def screenshot_path(self):
        if self._screenshot_path:
            return self._screenshot_path
        return os.path.join(self.main_path, 'screenshots', f'screen_{self.time_step}.png')
    
    @screenshot_path.setter
    def screenshot_path(self, value):
        self._screenshot_path = value

    @property
    def trace_path(self):
        return os.path.join(self.main_path, 'playwright_traces', f'{self.time_step}.zip')

    @property
    def dom_tree_path(self):
        return os.path.join(self.main_path, 'dom', f'{self.time_step}.html')
    
    def normalize_coords(self, x, y, image_path):
        try:
            # Ensure x and y are numeric
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                self.logger.error(f"Invalid coordinate types: x={type(x)}, y={type(y)}")
                # Return center of viewport
                vs = getattr(self.page, 'viewport_size', None)
                if isinstance(vs, dict) and 'width' in vs and 'height' in vs:
                    viewport_width, viewport_height = vs['width'], vs['height']
                else:
                    viewport_width = self.config.get('browser', {}).get('viewport', {}).get('width', 1500)
                    viewport_height = self.config.get('browser', {}).get('viewport', {}).get('height', 1200)
                return (viewport_width // 2, viewport_height // 2)
            
            # Prefer real viewport from Playwright
            viewport_width = None
            viewport_height = None
            vs = getattr(self.page, 'viewport_size', None)
            if isinstance(vs, dict) and 'width' in vs and 'height' in vs:
                viewport_width, viewport_height = vs['width'], vs['height']
            
            # Final fallback: config viewport
            if viewport_width is None or viewport_height is None:
                viewport_width = self.config.get('browser', {}).get('viewport', {}).get('width', 1500)
                viewport_height = self.config.get('browser', {}).get('viewport', {}).get('height', 1200)
            
            scaled_x = float(x)
            scaled_y = float(y)
            
            # Clamp to viewport bounds and convert to int
            clamped_x = int(max(0, min(viewport_width - 1, scaled_x)))
            clamped_y = int(max(0, min(viewport_height - 1, scaled_y)))
            
            self.logger.debug(f"Clamped coords: input({x},{y}) → Pixels({clamped_x},{clamped_y})[{viewport_width}x{viewport_height}]")
            
            return (clamped_x, clamped_y)
            
        except Exception as e:
            self.logger.error(f"Coordinate normalization failed: {e}", exc_info=True)
            # Return safe center point as fallback
            vs = getattr(self.page, 'viewport_size', None)
            if isinstance(vs, dict) and 'width' in vs and 'height' in vs:
                viewport_width, viewport_height = vs['width'], vs['height']
            else:
                viewport_width = self.config.get('browser', {}).get('viewport', {}).get('width', 1500)
                viewport_height = self.config.get('browser', {}).get('viewport', {}).get('height', 1200)
            return (viewport_width // 2, viewport_height // 2)

    def map_normalized_to_pixels(self, x, y):
        try:
            vs = getattr(self.page, 'viewport_size', None)
            if isinstance(vs, dict) and 'width' in vs and 'height' in vs:
                viewport_width, viewport_height = vs['width'], vs['height']
            else:
                viewport_width = self.config.get('browser', {}).get('viewport', {}).get('width', 1500)
                viewport_height = self.config.get('browser', {}).get('viewport', {}).get('height', 1200)
            scaled_x = (float(x) / 1000.0) * viewport_width
            scaled_y = (float(y) / 1000.0) * viewport_height
            return (int(max(0, min(viewport_width - 1, scaled_x))), int(max(0, min(viewport_height - 1, scaled_y))))
        except Exception:
            vs = getattr(self.page, 'viewport_size', None)
            if isinstance(vs, dict) and 'width' in vs and 'height' in vs:
                viewport_width, viewport_height = vs['width'], vs['height']
            else:
                viewport_width = self.config.get('browser', {}).get('viewport', {}).get('width', 1500)
                viewport_height = self.config.get('browser', {}).get('viewport', {}).get('height', 1200)
            return (viewport_width // 2, viewport_height // 2)

    @property
    def accessibility_tree_path(self):
        return os.path.join(self.main_path, 'accessibility', f'{self.time_step}.json')

    async def _capture_page_state(self):
        return await capture_page_state(
            self.page,
            logger=self.logger,
            pending_hit_test_coords=getattr(self, "_pending_hit_test_coords", None)
        )

    async def _detect_page_state_change(self, state_before, state_after, action_type):
        return detect_page_state_change(
            state_before,
            state_after,
            action_type,
            logger=self.logger,
            state_store=self.__dict__
        )
    
    async def _find_click_target_by_text(self, text):
        return await find_click_target_by_text(self.page, text, logger=self.logger)
    
    async def _analyze_previous_action_results(self):
        return analyze_previous_action_results(self.taken_actions, logger=self.logger)

    def _add_action_to_stack(self, action_type, element_info=None, coordinates=None):
        """
        Add action to the action stack for repetition detection.
        
        Args:
            action_type (str): The type of action (CLICK, TYPE, etc.)
            element_info (str): Information about the target element
            coordinates (tuple): Click coordinates if applicable
        """
        self.action_stack = add_action_to_stack(
            self.action_stack,
            self.max_stack_size,
            len(self.taken_actions),
            action_type,
            element_info=element_info,
            coordinates=coordinates,
            logger=self.logger
        )

    def _detect_repetitive_actions(self):
        """
        Detect if recent actions are repetitive and should be forbidden.
        
        Returns:
            dict: Detection result with forbidden actions and suggestions
        """
        return detect_repetitive_actions(self.action_stack, self.forbidden_actions, self.taken_actions, logger=self.logger)

    def _is_action_forbidden(self, action_type, element_info=None, coordinates=None):
        """
        Check if a proposed action is forbidden due to repetition.
        
        Args:
            action_type (str): The proposed action type
            element_info (str): Element information
            coordinates (tuple): Coordinates if applicable
            
        Returns:
            bool: True if action is forbidden
        """
        return is_action_forbidden(
            self.forbidden_actions,
            action_type,
            element_info=element_info,
            coordinates=coordinates,
            logger=self.logger
        )

    def _manage_action_history(self):
        """
        Manage action history by limiting history length.
        This function should be called after each action execution.
        """
        self.taken_actions = manage_action_history(self.taken_actions, self.max_action_history, logger=self.logger)
