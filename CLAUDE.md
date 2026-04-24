# CrisisOps

Read prompt_v3.txt first. It is the authoritative spec.
Do not invent anything not in that file.
Do not simplify reward components or the candor system.

## Workflow

Run `python -m py_compile <file>` after each file to verify it compiles.
After calibration/calibrate.py is built, run it and paste results before continuing.
Grep for `candor` in agent observation code before moving past environment.py — it must never appear there.

## Status

Repo is empty. Build in the order specified at the bottom of prompt_v3.txt.
