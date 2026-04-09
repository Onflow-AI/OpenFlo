# OpenFlo <img src="onflow-logo.svg" align="right" width="140">

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)]() [![PDF](https://img.shields.io/badge/PDF-coming%20soon-red.svg)]()

*Developed under [Nexus Labs](https://github.com/UCL-Nexus-Labs) · University College London*

> Built upon [Avenir-Web](https://github.com/Princeton-AI2-Lab/Avenir-Web) by the Princeton AI² Lab.

**Authors:**
[Wee Joe Tan](mailto:joe.tan.25@ucl.ac.uk)\*†,
[Zi Rui Lucas Lim](mailto:zi.lim.25@ucl.ac.uk)\*†,
[Shashank Durgad](mailto:shashank.durgad.25@ucl.ac.uk)\*†,
[Karim Obegi](mailto:karim.obegi.25@ucl.ac.uk)\*†,
[Aiden Yiliu Li](mailto:yiliu.li.23@ucl.ac.uk)\*†‡

<sub>\* Equal contribution &nbsp;† University College London &nbsp;‡ Also affiliated with Princeton AI² Lab</sub>

---

## Abstract

OpenFlo is an autonomous web agent framework for systematic measurement of web usability patterns at scale. Extending the Avenir-Web architecture, OpenFlo introduces a multi-metric UX evaluation layer that scores each agent action across four dimensions — **overall ease (SEQ), efficiency, clarity, and confidence** — using the established Single Ease Question (SEQ) methodology extended beyond its original single-item form. Step-level scores are then synthesised into a **System Usability Scale (SUS)** report at session end, following the standard Brooke (1996) scoring formula and Sauro-Lewis curved grading. A configurable **persona framework** injects user archetypes (defined by digital literacy, device, reading speed, and friction tolerance) directly into LLM scoring prompts, enabling comparative usability analysis across demographic profiles without re-running tasks. Together these components form a fully automated pipeline for studying usability patterns across real websites at scale.

---

## News

*(No releases yet — check back soon.)*

---

## Installation

Requirements:
- Python `>=3.9` (3.11 recommended)
- Playwright-compatible browser (Chromium recommended)
- An API key for your chosen LLM provider (OpenRouter preferred)

```bash
conda create -n openflo python=3.11
conda activate openflo

pip install -e src

playwright install chromium
```

---

## API Keys

Recommended — set as an environment variable:

```bash
export OPENROUTER_API_KEY="your-key"
```

Or copy the template and fill it in:

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY
```

Environment variables take precedence over any `[api_keys]` values in the config TOML.

---

## Quickstart

Run all commands from `src/` (config paths are relative to `src/`):

```bash
cd src
```

### Batch run

Set `experiment.task_file_path` in `src/config/auto_mode.toml`, then:

```bash
uv run run_agent.py -c config/auto_mode.toml
```

### With a persona

```bash
uv run run_agent.py -c config/auto_mode.toml -p config/persona.toml
```

The persona is injected into SEQ scoring prompts and surfaced in the final `sus_report.json`. See [`src/config/persona.toml`](src/config/persona.toml) for all available fields with inline documentation.

### Task JSON format

Batch mode expects a JSON array:

```json
[
  {
    "task_id": "task_001",
    "confirmed_task": "Find the official API docs for X",
    "website": "https://example.com/"
  }
]
```

---

## UX Evaluation

OpenFlo's primary research contribution is an automated UX evaluation pipeline built on top of Avenir-Web's agent execution layer.

### Step-level scoring: extended SEQ

The [Single Ease Question (SEQ)](https://measuringu.com/seq10/) is a validated 7-point micro-usability metric for individual task steps. OpenFlo extends it with three additional dimensions, all scored 1–7 after each agent action:

| Metric | What it captures |
|---|---|
| **SEQ** (overall ease) | Perceived difficulty of completing the action |
| **Efficiency** | Whether the path to the action was direct and fast |
| **Clarity** | How understandable the UI element or system response was |
| **Confidence** | How certain the user felt about the action and its outcome |

Each metric also produces a 1–2 sentence qualitative assessment. Low-scoring steps (SEQ ≤ 3) are flagged as friction points and classified by severity.

### Session-level scoring: SUS

At session end, the [System Usability Scale (SUS)](https://measuringu.com/sus/) (Brooke, 1996) is computed from the accumulated step-level data using the standard formula:

```
Final SUS Score = (X + Y) × 2.5
  X = Σ (score − 1) for positive items  {1, 3, 5, 7, 9}
  Y = Σ (5 − score) for negative items  {2, 4, 6, 8, 10}
```

Grades follow the [Sauro-Lewis curved scale](https://measuringu.com/sus/) (A+ ≥ 84.1 → F < 51.7). A statistical heuristic fallback is used if the LLM is unavailable, using volatility and learning-curve analysis across the four metric dimensions.

To enable UX evaluation, set in your config:

```toml
[ux]
enable_synthesis = true
```

---

## Persona Framework

UX sessions can be evaluated through the lens of a configurable user persona. When a persona is active, the LLM embodies the described user's profile when scoring each step — adjusting judgements based on their characteristics:

| Field | Options |
|---|---|
| `digital_literacy` | `"expert"` \| `"intermediate"` \| `"beginner"` \| `"very_low"` |
| `primary_device` | `"desktop_keyboard"` \| `"desktop_mouse"` \| `"tablet_touch"` \| `"mobile_touch"` |
| `reading_speed` | `"fast"` \| `"normal"` \| `"slow"` |
| `tolerance_for_friction` | `"high"` \| `"medium"` \| `"low"` \| `"very_low"` |
| `prior_experience` | Free text fed directly into scoring prompts |
| `description` | 3–4 sentence narrative the LLM embodies during scoring |
| `common_friction_types` | Labels surfaced in the report (e.g. `waiting`, `confusion`, `searching`) |
| `[persona.scoring_bias]` | Integer offsets applied per metric after LLM response (e.g. `seq_modifier = -1`) |

This makes it possible to compare how the same workflow scores for an expert desktop user vs. a low-literacy mobile user without re-running the task.

---

## Repository Layout

```
OpenFlo/
├── src/
│   ├── openflo/
│   │   ├── agent/
│   │   │   ├── agent.py          # Central orchestrator: predict → execute → evaluate loop
│   │   │   ├── config.py         # Config loading and validation (TOML + env)
│   │   │   ├── executor.py       # Action dispatch: click, type, scroll, drag, …
│   │   │   ├── predictor.py      # LLM interaction, action prediction, history compression
│   │   │   ├── evaluation.py     # Task completion verification and termination logic
│   │   │   └── reporting.py      # Result serialisation and action summary generation
│   │   ├── browser/              # Playwright integration and browser state management
│   │   ├── llm/                  # LLM engine abstraction (via LiteLLM)
│   │   ├── managers/
│   │   │   └── ux_synthesis.py   # SEQ-to-SUS orchestration (UXSynthesisManager)
│   │   ├── ux/
│   │   │   ├── seq_scorer.py     # Multi-metric SEQ evaluator
│   │   │   ├── sus_calculator.py # SUS scoring with Sauro-Lewis grading + heuristic fallback
│   │   │   └── report_generator.py # Markdown and JSON UX report output
│   │   ├── personas/
│   │   │   └── profile.py        # PersonaProfile dataclass
│   │   ├── prompts/              # Prompt templates and builders
│   │   └── utils/                # Image processing, reasoning utilities
│   ├── run_agent.py              # Entry point: demo + batch runner
│   └── config/
│       ├── auto_mode.toml        # Primary config (model, playwright, UX, experiment)
│       └── persona.toml          # Example persona config with inline documentation
├── data/                         # Example task JSON files
└── .env.example                  # API key template
```

---

## Configuration

Configs are TOML files. See `src/config/auto_mode.toml` for a fully annotated example.

### `[basic]`
- `save_file_dir` — output root directory
- `default_task`, `default_website` — used when no batch file is provided

### `[model]`
- `name` — model identifier (e.g. `openrouter/anthropic/claude-sonnet-4-5`)
- `temperature`, `rate_limit`
- `reasoning_model` — separate model for termination/evaluation reasoning
- `checklist_model` — model for checklist management
- `completion_eval_model` — model for final task success evaluation

### `[experiment]`
- `task_file_path` — JSON task list for batch mode
- `overwrite` — skip or overwrite existing task output folders
- `max_op`, `max_continuous_no_op` — execution limits
- `highlight` — draw labeled overlays on screenshots

### `[playwright]`
- `headless`, `viewport`, `tracing`, `save_video`
- `locale`, `geolocation` — for locale/region-sensitive tasks

### `[ux]`
- `enable_synthesis` — enable SEQ/SUS evaluation (default `false`)
- `generate_report` — write `sus_report.json` at session end (default `true`)
- `ux_model` — model for SEQ scoring; falls back to main model if omitted
- `seq_screenshot_context` — include screenshots in step evaluation (default `true`)

### `[persona]`
All fields optional; or pass a separate file with `-p persona.toml`. See the `[Persona Framework](#persona-framework)` section for field definitions.

---

## Outputs

Each task writes to `<save_file_dir>/<task_id>/`:

| File | Contents |
|---|---|
| `agent.log` | Per-task execution log |
| `result.json` | Final summary (task, actions, outcome, timing) |
| `config.toml` | Resolved config snapshot |
| `all_predictions.json` | Full LLM I/O trace for the task |
| `screenshots/` | `screen_<step>.png` and `screen_<step>_labeled.png` |
| `sus_report.json` | UX evaluation: SEQ scores per step, SUS score and grade, friction analysis, persona context |

Run-level logs are written to `src/logs/`.

---

## Troubleshooting

- **Missing API key** — fill in `OPENROUTER_API_KEY` in `.env` (copy from `.env.example`)
- **Playwright browser not found** — run `playwright install chromium`
- **Want to watch the browser** — set `playwright.headless = false`
- **Config paths look wrong** — run from `src/` or pass an absolute path with `-c`

---

## References

- Brooke, J. (1996). SUS: A quick and dirty usability scale. In P. Jordan et al. (Eds.), *Usability Evaluation in Industry*. Taylor & Francis.
- Sauro, J. & Lewis, J. R. (2011). Correlations among prototypical usability metrics: evidence for the construct of usability. *CHI '11*, ACM. — basis for the Sauro-Lewis grading scale
- [measuringu.com/seq10/](https://measuringu.com/seq10/) — SEQ methodology and benchmarks
- [measuringu.com/sus/](https://measuringu.com/sus/) — SUS grading and percentile lookup

---

## Acknowledgement

OpenFlo is built upon [Avenir-Web](https://github.com/Princeton-AI2-Lab/Avenir-Web) by the Princeton AI² Lab. We thank the Avenir-Web authors for open-sourcing their framework.

---

## Disclaimer

This repository is provided for research use. Model outputs may be incorrect, incomplete, or unsafe. You are responsible for reviewing agent actions and complying with applicable laws and website terms of service when running web automation.

---

## Contact

Wee Joe Tan — [joe.tan.25@ucl.ac.uk](mailto:joe.tan.25@ucl.ac.uk)

---

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

Copyright © 2025 UCL Nexus Labs. All rights reserved.
