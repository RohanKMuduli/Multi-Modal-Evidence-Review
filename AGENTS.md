# Agent Instructions

- Keep source code under `code/`.
- Keep input datasets under `dataset/`.
- Do not commit generated outputs, logs, caches, archives, or local secret files.
- Do not commit `.env`; use `.env.example` or environment variables for API keys.
- Preserve the CSV schemas in `dataset/`.
- Run `python -m compileall -q code` before submission after code changes.
- Prefer `python code/main.py --mock-mode` for fast local checks.
