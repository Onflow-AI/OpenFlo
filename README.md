# OpenFlo

**An open-source web automation framework powered by large multimodal models.**

*Developed under [Nexus Labs](https://github.com/UCL-Nexus-Labs)*

> This project is based on [Avenir-Web](https://github.com/Princeton-AI2-Lab/Avenir-Web) developed by the Princeton AI2 Lab.

OpenFlo enables autonomous web agents to perform tasks on any website using vision-language models. The system combines robust browser automation with intelligent action prediction to execute complex workflows.

## Repository Layout

- [`src/openflo/`](src/openflo/): core agent implementation (`OpenFloAgent`)
  - [`agent/`](src/openflo/agent/): main agent logic
    - [`agent.py`](src/openflo/agent/agent.py): central agent class and execution flow
    - [`config.py`](src/openflo/agent/config.py): configuration loading and validation
    - [`reporting.py`](src/openflo/agent/reporting.py): result saving and summary generation
    - [`evaluation.py`](src/openflo/agent/evaluation.py): task success evaluation and termination logic
    - [`executor.py`](src/openflo/agent/executor.py): action execution logic
    - [`predictor.py`](src/openflo/agent/predictor.py): LLM interaction and action prediction
  - [`managers/ux_synthesis.py`](src/openflo/managers/ux_synthesis.py): SEQ-to-SUS evaluation orchestration
  - [`ux/`](src/openflo/ux/): UX scoring components (SEQ scorer, SUS calculator, report generator)
  - [`personas/profile.py`](src/openflo/personas/profile.py): `PersonaProfile` dataclass for persona-biased evaluation
- [`src/run_agent.py`](src/run_agent.py): single-process runner (demo + batch)
- [`src/config/`](src/config/)*.toml: sample configs (including `persona.toml`)
- [`data/`](data/): example data and task files

## Requirements

- Python `>=3.9` (`src/pyproject.toml`)
- A browser for Playwright (Chromium recommended)
- An API key for your chosen provider (OpenRouter preferred)

## Setup

From the repository root:

```bash
# Create a conda environment
conda create -n openflo python=3.11
conda activate openflo

# Install the package in editable mode
pip install -e src

# Set up Playwright and install browser kernels
playwright install
```

Set your API key in the `.env` file at the project root:

```bash
cp .env.example .env
# then edit .env and fill in your key
```

Environment variables take precedence over anything in `[api_keys]` inside the config.

## Running

Run scripts from `src/` (paths in configs are written relative to `src/`):

```bash
cd src
```

### Batch Mode (JSON list of tasks)

In your config (`src/config/auto_mode.toml`), set `experiment.task_file_path`, then run:

```bash
uv run run_agent.py -c config/auto_mode.toml
```

### With a Persona

Pass a persona config with `-p` to bias UX evaluation from a specific user's perspective:

```bash
uv run run_agent.py -c config/auto_mode.toml -p config/persona.toml
```

The persona is injected into the final UX report — scores are evaluated as if the described user performed the task. See [`src/config/persona.toml`](src/config/persona.toml) for all available fields and inline documentation.

## Task JSON Format

Batch mode expects a JSON array of tasks like:

```json
[
  {
    "task_id": "task_001",
    "confirmed_task": "Find the official API docs for X",
    "website": "https://example.com/"
  }
]
```

## Configuration Overview

Configs are TOML files; see `src/config/auto_mode.toml`.

- `[basic]`
  - `save_file_dir`: output root directory
  - `default_task`, `default_website`: defaults for single-task runs
- `[experiment]`
  - `task_file_path`: JSON tasks list for batch mode
  - `overwrite`: skip or overwrite existing task output folders
  - `max_op`, `max_continuous_no_op`, `highlight`
- `[model]`
  - `name`: model identifier (commonly `openrouter/...`)
  - `temperature`, `rate_limit`
  - optional: `reasoning_model`, `checklist_model`, `completion_eval_model`
- `[api_keys]`
  - keys are loaded from `.env` (`OPENROUTER_API_KEY`)
  - individual keys can be uncommented in the toml to override
- `[playwright]`
  - `headless`, `viewport`, `tracing`, `save_video`, `locale`, `geolocation`
- `[ux]` *(optional)*
  - `enable_synthesis`: enable SEQ/SUS evaluation (default `false`)
  - `generate_report`: write `sus_report.json` at session end (default `true`)
  - `ux_model`: model for SEQ scoring — defaults to main model if omitted
  - `seq_screenshot_context`: include screenshots in step evaluation (default `true`)
- `[persona]` *(optional — or pass via `-p persona.toml`)*
  - `id`, `display_name`, `age_range`: identification fields
  - `digital_literacy`: `"expert"` | `"intermediate"` | `"beginner"` | `"very_low"`
  - `primary_device`: `"desktop_keyboard"` | `"desktop_mouse"` | `"tablet_touch"` | `"mobile_touch"`
  - `reading_speed`: `"fast"` | `"normal"` | `"slow"`
  - `tolerance_for_friction`: `"high"` | `"medium"` | `"low"` | `"very_low"`
  - `prior_experience`: free text fed to the LLM
  - `description`: 3–4 sentence narrative the LLM embodies when scoring
  - `common_friction_types`: list of friction labels surfaced in the report
  - `[persona.scoring_bias]`: integer offsets applied to metric scores after LLM response (e.g. `seq_modifier = -1`)

## Outputs

Each task writes to `basic.save_file_dir/<task_id>/`:

- `agent.log`: per-task execution log
- `result.json`: final summary (handled by `src/openflo/agent/reporting.py`)
- `config.toml`: resolved config snapshot
- `all_predictions.json`: recorded LLM I/O for the task
- `screenshots/`: `screen_<step>.png` and sometimes `screen_<step>_labeled.png`
- `sus_report.json`: UX evaluation report (SEQ scores, SUS score, friction points, persona context) — written when `ux.enable_synthesis = true`

The runners also write run-level logs to `src/logs/`.

## Troubleshooting

- Missing API key: fill in `OPENROUTER_API_KEY` in `.env` (copy from `.env.example`)
- Playwright browser not found: run `uv run playwright install chromium`
- Want to watch the browser: set `playwright.headless = false`
- Config paths look wrong: run from `src/` or pass an absolute `-c` config path

## Attribution

OpenFlo is built upon [Avenir-Web](https://github.com/w3joe/Avenir-Web).





## License

This project maintains the same license as the original framework. See the [LICENSE](LICENSE) file for details.
