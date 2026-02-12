# -*- coding: utf-8 -*-
import os
import toml

def load_agent_config(
    config_path=None,
    config=None,
    save_file_dir="seeact_agent_files",
    default_task='Find the pdf of the paper "GPT-4V(ision) is a Generalist Web Agent, if Grounded"',
    default_website="https://www.google.com/",
    input_info=["screenshot"],
    crawler_mode=False,
    crawler_max_steps=20,
    max_auto_op=30,
    max_continuous_no_op=15,
    highlight=False,
    headless=False,
    args=[],
    browser_app="chrome",
    persistant=False,
    persistant_user_path="",
    save_video=False,
    viewport={"width": 1280, "height": 720},
    stealth_mode=True,
    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0",
    tracing=False,
    trace={"screenshots": True, "snapshots": True, "sources": True},
    model="openrouter/qwen/qwen-2.5-72b-instruct",
    temperature=1.0
):
    """
    Load and merge configuration from file, dict, or defaults.
    """
    try:
        if config is not None:
            # If config dictionary is passed directly, use it
            pass
        elif config_path is not None:
            with open(config_path, 'r') as config_file:
                print(f"Configuration File Loaded - {config_path}")
                config = toml.load(config_file)
        else:
            config = {
                "basic": {
                    "save_file_dir": save_file_dir,
                    "default_task": default_task,
                    "default_website": default_website,
                    "crawler_mode": crawler_mode,
                    "crawler_max_steps": crawler_max_steps,
                },
                "agent": {
                    "input_info": input_info,
                    "max_auto_op": max_auto_op,
                    "max_continuous_no_op": max_continuous_no_op,
                    "highlight": highlight
                },
                "model": {
                    "name": model,
                    "temperature": temperature
                }
            }
        
        # Update with runtime overrides
        config.update({
            "browser": {
                "headless": headless,
                "args": args,
                "browser_app": browser_app,
                "persistant": persistant,
                "persistant_user_path": persistant_user_path,
                "save_video": save_video,
                "viewport": viewport,
                "tracing": tracing,
                "trace": trace,
                # Simple anti-detection settings
                "stealth_mode": stealth_mode,
                "user_agent": user_agent
            }
        })
        
        return config

    except FileNotFoundError:
        print(f"Error: File '{os.path.abspath(config_path)}' not found.")
        return None
    except toml.TomlDecodeError:
        print(f"Error: File '{os.path.abspath(config_path)}' is not a valid TOML file.")
        return None
