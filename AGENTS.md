# Repository Guidelines

## Project Structure & Module Organization
- `modeling/` hosts the unified MoT architecture, including `bagel/`, `qwen2/`, `siglip/`, and `autoencoder.py`.
- `train/` contains stage-specific entry points such as `pretrain_unified_navit.py` and supporting configs under `data/configs/`.
- `scripts/` provides automation: `train.sh` for distributed torchrun, `eval/` for benchmark wrappers, and `download_model.py` to fetch weights.
- `eval/` embeds benchmark integrations (VLM, GenEval, WISE, KRIS) while `results/` is the default logging root.
- `app.py` and `inferencer.py` expose interactive demos; assets and sample inputs live in `assets/`, `test_images/`, and `models/`.

## Build, Test, and Development Commands
- `conda create -n bagel python=3.10 -y && conda activate bagel` to match training recipes.
- `pip install -r requirements.txt` installs core runtime libraries; add `flash_attn==2.5.8 --no-build-isolation` for GPU inference.
- `python app.py --mode 2 --zh` starts the Gradio UI with NF4 quantization, while `python inferencer.py` can be scripted for batch runs.
- `bash scripts/train.sh` acts as the baseline distributed launcher; replace the exported variables before execution.

## Coding Style & Naming Conventions
Python modules use 4-space indentation, type hints, and descriptive class names (e.g., `InterleaveInferencer`). Align new configs with the hyphenated snake_case patterns already in `data/configs/example.yaml`. Keep checkpoints and assets in snake_case directories. Run `python -m compileall .` or formatters such as `ruff`/`black` if you add them, and document any new tooling.

## Testing Guidelines
Primary validation is benchmark-driven. Run `bash scripts/eval/run_eval_vlm.sh` for understanding tasks, `run_geneval.sh` for text-to-image, and `run_kris.sh` or `run_wise.sh` for advanced reasoning. Capture metrics under `results/` and attach key logs to PRs. For new data flows, add minimal smoke scripts under `eval/` or `scripts/` and reference sample prompts in `eval/gen/`.

## Commit & Pull Request Guidelines
Recent commits are short, imperative statements (e.g., “add taylorseer support”). Follow that style and keep related changes grouped. For PRs, include: objective summary, modified configs, command lines used, links to datasets or checkpoints, and table snippets or screenshots for visual tasks. Flag any required credentials (OpenAI tokens, Hugging Face access) and note reproducibility steps for reviewers.

## Security & Configuration Tips
Store API keys as environment variables before invoking evaluation scripts (`export openai_api_key=...`). Avoid checking large checkpoints into Git; stage them under `models/` with `.gitignore`. When sharing logs, scrub user data and confirm external URLs are accessible to reviewers.
