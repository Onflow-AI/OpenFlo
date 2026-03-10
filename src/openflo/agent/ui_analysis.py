import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

ALLOWED_CATEGORIES = {
    "layout",
    "color_contrast",
    "text_readability",
    "button_usability",
    "learnability",
}

def clamp_rating(v):
    try:
        v = int(v)
    except:
        v = 0
    return max(0, min(11, v))

def _extract_json(text: str) -> Optional[str]:
    """Extract the first JSON object from a model response."""
    if not text:
        return None
    # Remove ```json fences
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Try to find the first {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0) if m else None

def _clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(round(float(x)))
        return max(lo, min(hi, v))
    except Exception:
        return default

def _normalize_bbox(b: Any, w: int, h: int) -> Optional[List[int]]:
    """
    Expect bbox [x1,y1,x2,y2] in pixels. Clamp to image bounds.
    """
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    x1 = _clamp_int(b[0], 0, w - 1, 0)
    y1 = _clamp_int(b[1], 0, h - 1, 0)
    x2 = _clamp_int(b[2], 0, w - 1, 0)
    y2 = _clamp_int(b[3], 0, h - 1, 0)
    # Ensure proper ordering
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    # Minimum non-zero area is optional; keep as-is
    return [x1, y1, x2, y2]

def _validate_ui_analysis(obj: Any, frame: int, image_size: Tuple[int, int]) -> Dict[str, Any]:
    w, h = image_size
    out: Dict[str, Any] = {
        "frame": frame,
        "image_size": [w, h],
        "ratings": {
            "aesthetics": 0,
            "usability": 0,
            "overall_design": 0
        },
        "critiques": []
    }

    if not isinstance(obj, dict):
        return out

    ratings = obj.get("ratings", {})
    if isinstance(ratings, dict):
        out["ratings"]["aesthetics"] = _clamp_int(ratings.get("aesthetics"), 0, 11, 0)
        out["ratings"]["usability"] = _clamp_int(ratings.get("usability"), 0, 11, 0)
        out["ratings"]["overall_design"] = _clamp_int(ratings.get("overall_design"), 0, 11, 0)

    critiques = obj.get("critiques", [])
    if isinstance(critiques, list):
        cleaned: List[Dict[str, Any]] = []
        for c in critiques:
            if not isinstance(c, dict):
                continue
            cat = str(c.get("category") or "").strip()
            if cat not in ALLOWED_CATEGORIES:
                continue
            comment = str(c.get("comment") or "").strip()
            bbox = _normalize_bbox(c.get("bbox"), w, h)
            if not comment:
                continue
            if bbox is None:

                cleaned.append({"category": cat, "comment": comment, "bbox": None})
            else:
                cleaned.append({"category": cat, "comment": comment, "bbox": bbox})

        out["critiques"] = cleaned

    return out

def _build_ui_analysis_prompt(frame: int, image_size: Tuple[int, int]) -> List[str]:
    w, h = image_size

    system = (
        "You are a UI evaluation assistant. "
        "Return ONLY valid JSON (no markdown, no extra text)."
    )

    user = f"""
Analyze the provided UI screenshot (frame={frame}, image_size={w}x{h}) and output UI evaluation metrics.

You MUST return a JSON object with this exact schema:


{{
  "frame": {frame},
  "image_size": [{w}, {h}],
  "ratings": {{
    "aesthetics": 0-11,
    "usability": 0-11,
    "overall_design": 0-11
  }},
  "critiques": [
    {{
      "category": "layout",
      "comment": "...",
      "bbox": [x1, y1, x2, y2]
    }},
    {{
      "category": "color_contrast",
      "comment": "...",
      "bbox": [x1, y1, x2, y2]
    }},
    {{
      "category": "text_readability",
      "comment": "...",
      "bbox": [x1, y1, x2, y2]
    }},
    {{
      "category": "button_usability",
      "comment": "...",
      "bbox": [x1, y1, x2, y2]
    }},
    {{
      "category": "learnability",
      "comment": "...",
      "bbox": [x1, y1, x2, y2]
    }}
  ]
}}

Rules:
- Ratings must be integers in [0, 11].
- bbox coordinates must be pixel coordinates within the image bounds.
- Provide 1-5 critiques max. Only include issues that are visible.
- If you cannot find issues for a category, omit that category (do NOT invent).
- Return JSON ONLY.
""".strip()

    # engine.generate expects [prompt0, prompt1, prompt2]
    return [system, user, ""]

async def run_ui_analysis(
    agent,
    frame: int,
    image_paths: List[str],
    image_size: Tuple[int, int] = (1400, 1400),
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Runs per-frame UI analysis on screen_{t}.png using agent.engine.generate (OpenRouter multimodal).
    Returns validated dict or None.
    """
    if not image_paths:
        return None
    img_path = image_paths[0]
    if not os.path.exists(img_path):
        return None

    prompt = _build_ui_analysis_prompt(frame, image_size)

    # Let caller override model; otherwise use agent default model.
    current_model = model

    try:
        resp = await agent.engine.generate(
            prompt=prompt,
            temperature=0.0,
            max_new_tokens=800,
            turn_number=0,
            image_path=img_path,
            model=current_model,
        )
    except Exception as e:
        agent.logger.warning(f"UI analysis LLM call failed: {e}")
        return None

    raw_text = resp if isinstance(resp, str) else str(resp)

    json_text = _extract_json(raw_text)
    if not json_text:
        agent.logger.warning("UI analysis: no JSON found in response")
        return None

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        agent.logger.warning(f"UI analysis: invalid JSON. Raw: {raw_text[:300]}")
        return None

    return _validate_ui_analysis(parsed, frame=frame, image_size=image_size)