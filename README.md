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
- [`src/run_agent.py`](src/run_agent.py): single-process runner (demo + batch)
- [`src/config/`](src/config/)*.toml: sample configs
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
python run_agent.py -c config/auto_mode.toml
```

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

## Outputs

Each task writes to `basic.save_file_dir/<task_id>/`:

- `agent.log`: per-task execution log
- `result.json`: final summary (handled by `src/openflo/agent/reporting.py`)
- `config.toml`: resolved config snapshot
- `all_predictions.json`: recorded LLM I/O for the task
- `screenshots/`: `screen_<step>.png` and sometimes `screen_<step>_labeled.png`

The runners also write run-level logs to `src/logs/`.

## Troubleshooting

- Missing API key: fill in `OPENROUTER_API_KEY` in `.env` (copy from `.env.example`)
- Playwright browser not found: run `python -m playwright install chromium`
- Want to watch the browser: set `playwright.headless = false`
- Config paths look wrong: run from `src/` or pass an absolute `-c` config path

## Attribution

OpenFlo is built upon [Avenir-Web](https://github.com/w3joe/Avenir-Web).





## License

This project maintains the same license as the original framework. See the [LICENSE](LICENSE) file for details.
