# CrisisOps

Read instructions.txt first. It is the authoritative spec.
Do not invent anything not in that file.
Do not simplify reward components or the candor system.
Run `python -m py_compile <file>` after each file to verify it compiles.

## Session Checkpoints

After each file, run `python -m py_compile <file>` and fix errors before continuing.
After environment.py: run a manual reset()/step() smoke test.
After calibrate.py: run it and paste results back before continuing.
Cross-file invariant: candor float is NEVER in agent observation — grep for it before moving on.
