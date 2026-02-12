"""
Reasoning utilities for task planning using advanced models.
"""

import os
import litellm
import asyncio
from seeact.llm.engine import add_llm_io_record
from seeact.prompts.templates import build_reasoning_prompt, format_reasoning_for_prompt


async def generate_task_reasoning(task_description: str, website: str = None,
                                  model: str = "anthropic/claude-sonnet-4.5", 
                                  enable_thinking: bool = True, enable_online: bool = True,
                                  reasoning_effort: str = "high",
                                  reasoning_verbosity: str = "high",
                                  use_web_search: bool = False,
                                  temperature: float = 1.0, logger=None,
                                  policy_constraints: str = "",
                                  plugins: list = None,
                                  task_id: str = None) -> dict:
    """
    Generate strategic reasoning for a web automation task.
    
    Args:
        task_description: The task to accomplish
        website: Target website URL (optional but recommended)
        model: Model to use (with or without openrouter/ prefix)
        enable_thinking: Reserved for future thinking-mode models
        enable_online: Reserved for future online-search models
        temperature: Sampling temperature
        logger: Optional logger
        
    Returns:
        dict with keys: reasoning (str), success (bool), error (str or None)
    """
    try:
        # Build web-automation specific prompt
        system_prompt, user_prompt = build_reasoning_prompt(task_description, website, policy_constraints)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Ensure openrouter/ prefix
        api_model = model if model.startswith("openrouter/") else f"openrouter/{model}"
        if enable_online and api_model.startswith("openrouter/") and not api_model.endswith(":online"):
            api_model = f"{api_model}:online"
        
        # Enforce GPT-5 usage and forbid o1-family
        if "o1" in api_model.lower():
            raise ValueError("O1 models are not permitted for reasoning. Please configure GPT-5 instead.")
        
        if logger:
            logger.info(f"🧠 Generating task reasoning with {api_model}")
        
        # Do not send tools or extra_body for OpenRouter – match other calls
        
        # Call LiteLLM
        call_params = {
            "model": api_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 300,
            "api_key": os.getenv("OPENROUTER_API_KEY"),
        }
        if enable_online and plugins:
            call_params["extra_body"] = {"plugins": plugins}
        
        # Keep request minimal to match other OpenRouter usage in repo
        
        response = await litellm.acompletion(**call_params)
        reasoning_text = response.choices[0].message.content or ""
        reasoning_text = reasoning_text.strip()
        import re
        m = re.search(r"<plan>([\s\S]*?)</plan>", reasoning_text, re.IGNORECASE)
        if not m:
            repair_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + "\n\nReturn ONLY <plan>...</plan> with the comprehensive mindmap."}
            ]
            try:
                repair_params = dict(call_params)
                repair_params["messages"] = repair_messages
                repair_params["temperature"] = 0.0
                repair_params["max_tokens"] = 220
                repair_resp = await litellm.acompletion(**repair_params)
                repair_text = (repair_resp.choices[0].message.content or "").strip()
                m2 = re.search(r"<plan>([\s\S]*?)</plan>", repair_text, re.IGNORECASE)
                if m2:
                    reasoning_text = m2.group(1).strip()
            except Exception:
                pass
        else:
            reasoning_text = m.group(1).strip()
        if logger:
            logger.info(f"✅ Generated reasoning: {reasoning_text}")
        try:
            add_llm_io_record({
                "model": api_model,
                "turn_number": 0,
                "messages": messages,
                "image_paths": None,
                "output": reasoning_text,
                "context": "task_reasoning",
                "task_id": task_id
            })
        except Exception:
            pass
        return {
            "reasoning": reasoning_text,
            "success": bool(reasoning_text),
            "error": None if reasoning_text else "Empty response"
        }
            
    except Exception as e:
        if logger:
            logger.error(f"❌ Reasoning generation failed: {e}")
        return {
            "reasoning": "",
            "success": False,
            "error": str(e)
        }


