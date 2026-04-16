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

from playwright.async_api import Playwright, async_playwright
from pathlib import Path
import toml
import os
import time
import logging
import json

async def normal_launch_async(playwright: Playwright, headless=False, args=None, channel=None):
    default_args = [
        "--disable-blink-features=AutomationControlled",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--disable-features=BackForwardCache,Translate",
    ]
    if args is None:
        args = default_args
    else:
        merged_extra = [a for a in default_args if a not in args]
        args = args + merged_extra
    
    browser = await playwright.chromium.launch(
        traces_dir=None,
        headless=headless,
        args=args,
        channel=channel,
    )
    return browser



async def normal_new_context_async(
        browser,
        storage_state=None,
        har_path=None,
        video_path=None,
        tracing=False,
        trace_screenshots=False,
        trace_snapshots=False,
        trace_sources=False,
        locale=None,
        geolocation=None,
        user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0",
        viewport: dict = {"width": 1280, "height": 720},
):
    context = await browser.new_context(
        storage_state=storage_state,
        user_agent=user_agent,
        viewport=viewport,
        device_scale_factor=1,
        locale=locale,
        record_har_path=har_path,
        record_video_dir=video_path,
        geolocation=geolocation,
        extra_http_headers={
            "Referer": "https://www.google.com/",
            "Accept-Language": "en-us"
        }
    )

    if tracing:
        await context.tracing.start(screenshots=trace_screenshots, snapshots=trace_snapshots, sources=trace_sources)
    return context


def saveconfig(config, save_file):
    """
    config is a dictionary.
    save_path: saving path include file name.
    """


    if isinstance(save_file, str):
        save_file = Path(save_file)
    if isinstance(config, dict):
        with open(save_file, 'w') as f:
            # Create a deep copy to avoid modifying the original config
            import copy
            config_without_key = copy.deepcopy(config)
            # Remove API key from api_keys section if it exists
            if "api_keys" in config_without_key and "openrouter_api_key" in config_without_key["api_keys"]:
                config_without_key["api_keys"]["openrouter_api_key"] = "Your API key here"
            toml.dump(config_without_key, f)
    else:
        os.system(" ".join(["cp", str(config), str(save_file)]))


def setup_agent_logger(task_id, main_path, redirect_to_dev_log=False):
    logger_name = f"{task_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    logger.handlers.clear()
    
    log_filename = 'agent.log'
    f_handler = logging.FileHandler(os.path.join(main_path, log_filename), encoding='utf-8')
    f_handler.setLevel(logging.INFO)

    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)

    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')

    f_handler.setFormatter(file_formatter)
    c_handler.setFormatter(console_formatter)

    try:
        root_log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'log'))
        g_handler = logging.FileHandler(root_log_path, encoding='utf-8')
        g_handler.setLevel(logging.INFO)
        g_handler.setFormatter(file_formatter)
        logger.addHandler(g_handler)
    except Exception:
        pass

    logger.addHandler(f_handler)
    if not redirect_to_dev_log:
        logger.addHandler(c_handler)

    logger.propagate = False
    return logger


async def page_on_close_handler(agent):
    try:
        if getattr(agent, 'is_stopping', False):
            return
        if not (agent.session_control and isinstance(agent.session_control, dict)):
            return
        context = agent.session_control.get('context')
        browser = agent.session_control.get('browser')
        agent.logger.info("The active tab was closed. Will recover a working page.")
        if context:
            try:
                pages = context.pages if hasattr(context, 'pages') else []
            except Exception:
                pages = []
            if pages:
                try:
                    agent.page = pages[-1]
                    await agent.page.bring_to_front()
                    agent.logger.info(f"Switched the active tab to: {agent.page.url}")
                    return
                except Exception:
                    pass
            try:
                agent.page = await context.new_page()
            except Exception as e:
                agent.logger.warning(f"Context.new_page failed: {e}")
                if browser:
                    try:
                        geo_cfg = (
                            agent.config.get('browser', {}).get('geolocation') or
                            agent.config.get('playwright', {}).get('geolocation')
                        )
                        agent.session_control['context'] = await normal_new_context_async(
                            browser,
                            viewport=agent.config['browser']['viewport'],
                            user_agent=agent.config['browser']['user_agent'],
                            geolocation=geo_cfg
                        )
                        agent.session_control['context'].on("page", agent.page_on_open_handler)
                        agent.page = await agent.session_control['context'].new_page()
                    except Exception as e2:
                        agent.logger.error(f"Failed to recreate context: {e2}")
                        return
            try:
                target_url = getattr(agent, 'actual_website', None) or agent.config.get("basic", {}).get("default_website") or "https://www.google.com/"
                await agent.page.goto(target_url, wait_until="load")
                agent.logger.info(f"Switched the active tab to: {agent.page.url}")
            except Exception as e:
                agent.logger.info(f"Failed to navigate after refresh: {e}")
        else:
            if browser:
                try:
                    geo_cfg = (
                        agent.config.get('browser', {}).get('geolocation') or
                        agent.config.get('playwright', {}).get('geolocation')
                    )
                    agent.session_control['context'] = await normal_new_context_async(
                        browser,
                        viewport=agent.config['browser']['viewport'],
                        user_agent=agent.config['browser']['user_agent'],
                        geolocation=geo_cfg
                    )
                    agent.session_control['context'].on("page", agent.page_on_open_handler)
                    agent.page = await agent.session_control['context'].new_page()
                    target_url = getattr(agent, 'actual_website', None) or agent.config.get("basic", {}).get("default_website") or "https://www.google.com/"
                    await agent.page.goto(target_url, wait_until="load")
                    agent.logger.info(f"Recovered by creating new context: {agent.page.url}")
                except Exception as e:
                    agent.logger.error(f"Failed to recover without context: {e}")
                    return
    except Exception as e:
        agent.logger.warning(f"page_on_close_handler failed: {e}")


def save_action_history(main_path, taken_actions, logger, filename="action_history.txt"):
    history_path = os.path.join(main_path, filename)
    with open(history_path, 'w') as f:
        for action in taken_actions:
            f.write(action + '\n')
    logger.info(f"Action history saved to: {history_path}")


async def page_on_navigation_handler(frame):
    current_url = frame.url
    current_time = time.time()


async def page_on_crash_handler(agent, page):
    try:
        if getattr(agent, 'is_stopping', False):
            return
        crashed_url = None
        try:
            crashed_url = page.url
        except Exception:
            crashed_url = None
        agent.logger.error(f"Page crashed: {crashed_url or 'unknown URL'}")
        try:
            await page.close()
        except Exception:
            pass
        context = agent.session_control.get('context') if isinstance(agent.session_control, dict) else None
        if not context:
            agent.logger.error("No browser context available for crash handling")
            return
        new_page = await context.new_page()
        await agent.page_on_open_handler(new_page)
        target_url = None
        if crashed_url and not str(crashed_url).startswith(('about:', 'data:', 'chrome-error://', 'blob:')):
            target_url = crashed_url
        elif getattr(agent, 'actual_website', None):
            target_url = agent.actual_website
        if target_url:
            try:
                await new_page.goto(target_url, wait_until="domcontentloaded")
                agent.logger.info(f"Recovered to {target_url}")
            except Exception as e:
                agent.logger.warning(f"Failed to restore to {target_url}: {e}")
        else:
            agent.logger.info("Created a fresh page without navigation target")
        agent.page = new_page
    except Exception as e:
        agent.logger.error(f"Crash handling failed: {e}")


async def page_on_response_handler(agent, response):
    if agent.session_control and isinstance(agent.session_control, dict):
        agent.session_control['last_response'] = {
            "url": response.url,
            "status": response.status,
            "status_text": response.status_text,
            "headers": dict(response.headers)
        }


async def page_on_open_handler(agent, page):
    await page.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")
    await page.add_init_script("""
        (function(){
            try{
                var origClose = window.close;
                Object.defineProperty(window, 'close', { configurable: true, writable: true, value: function(){ try{ console.log('[Automation] window.close intercepted'); }catch(e){} } });
                window.addEventListener('beforeunload', function(){ try{ console.log('[Automation] beforeunload intercepted'); }catch(e){} });
            }catch(e){}
        })();
    """)
    page.on("close", agent.page_on_close_handler)
    page.on("crash", agent.page_on_crash_handler)
    page.on("response", agent.page_on_response_handler)
    agent.page = page


def get_page(agent):
    if agent._page is None:
        if (agent.session_control and isinstance(agent.session_control, dict) and 
            agent.session_control.get('active_page')):
            agent._page = agent.session_control['active_page']
    return agent._page


def set_page(agent, value):
    agent._page = value


async def start_agent_browser(agent, headless=None, args=None, website=None):
    agent.actual_website = website if website is not None else agent.config.get("basic", {}).get("default_website", "Unknown website")
    
    await agent._generate_task_reasoning()
    
    agent.playwright = await async_playwright().start()
    agent.session_control = {}
    agent.session_control['browser'] = await normal_launch_async(
        agent.playwright,
        headless=agent.config['browser']['headless'] if headless is None else headless,
        args=agent.config['browser']['args'] if args is None else args,
        channel=agent.config['browser'].get('browser_app', 'chrome')
    )
    geo_cfg = (
        agent.config.get('browser', {}).get('geolocation') or
        agent.config.get('playwright', {}).get('geolocation')
    )
    agent.session_control['context'] = await normal_new_context_async(
        agent.session_control['browser'],
        viewport=agent.config['browser']['viewport'],
        user_agent=agent.config['browser']['user_agent'],
        geolocation=geo_cfg
    )

    agent.session_control['context'].on("page", agent.page_on_open_handler)
    page = await agent.session_control['context'].new_page()
    await agent.page_on_open_handler(page)

    if agent.config["basic"].get("crawler_mode", False) is True:
        await agent.session_control['context'].tracing.start(screenshots=True, snapshots=True)

    if website is not None:
        try:
            await agent.page.goto(website, wait_until="load")
            agent.logger.info(f"Loaded website: {website}")
            
            if await agent._is_page_blocked_or_blank():
                agent.logger.error("⛔ Website is blocked, blank, or failed to load properly")
                agent.complete_flag = True
                return
        except Exception as e:
            agent.logger.error(f"Failed to load website: {e}")
            agent.logger.error("⛔ Terminating due to page load failure")
            agent.complete_flag = True
            return
    else:
        agent.logger.info("Browser started without initial navigation. Use GOTO action to navigate to a website.")

    if agent.tasks and not agent.checklist_generated:
        task_description = agent.tasks[-1] if isinstance(agent.tasks[-1], str) else str(agent.tasks[-1])
        agent.logger.info(f"Generating checklist for task: {task_description}")
        try:
            await agent.generate_task_checklist(task_description)
            agent.logger.info("Checklist generation completed successfully")
        except Exception as e:
            agent.logger.error(f"Checklist generation failed: {e}")
            agent.checklist_manager.task_checklist = [
                {"id": "execute", "description": f"Execute task: {task_description[:50]}...", "status": "pending"},
                {"id": "complete", "description": "Complete the task", "status": "pending"}
            ]
            agent.checklist_manager.checklist_generated = True
            agent.checklist_generated = True
            agent.logger.info("Created fallback checklist, continuing execution")


async def stop_agent_browser(agent):
    agent.is_stopping = True
    try:
        close_context = None
        try:
            if isinstance(agent.session_control, dict):
                close_context = agent.session_control.get('context', None)
            else:
                close_context = None
        except Exception:
            close_context = None
        if close_context:
            await close_context.close()
            agent.logger.info("Browser context closed.")
        if isinstance(agent.session_control, dict):
            agent.session_control['context'] = None
    except Exception as e:
        agent.logger.warning(f"Error closing browser context: {e}")
    
    try:
        if hasattr(agent, 'playwright') and agent.playwright:
            await agent.playwright.stop()
            agent.logger.info("Playwright instance stopped.")
            agent.playwright = None
    except Exception as e:
        agent.logger.warning(f"Error stopping playwright instance: {e}")


async def start_playwright_tracing(agent):
    if (agent.session_control and isinstance(agent.session_control, dict) and 
        agent.session_control.get('context') and 
        hasattr(agent.session_control['context'], 'tracing')):
        await agent.session_control['context'].tracing.start_chunk(
            title=f'Step-{agent.time_step}',
            name=f"{agent.time_step}"
        )


async def stop_playwright_tracing(agent):
    if (agent.session_control and isinstance(agent.session_control, dict) and 
        agent.session_control.get('context') and 
        hasattr(agent.session_control['context'], 'tracing')):
        await agent.session_control['context'].tracing.stop_chunk(path=agent.trace_path)


async def save_traces(agent):
    dom_tree = await agent.page.evaluate("document.documentElement.outerHTML")
    os.makedirs(os.path.join(agent.main_path, 'dom'), exist_ok=True)
    with open(agent.dom_tree_path, 'w', encoding='utf-8') as f:
        f.write(dom_tree)

    accessibility_tree = await agent.page.accessibility.snapshot()
    os.makedirs(os.path.join(agent.main_path, 'accessibility'), exist_ok=True)
    with open(agent.accessibility_tree_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(accessibility_tree, indent=4))
