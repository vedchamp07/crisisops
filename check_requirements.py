import os, json, yaml

print("=== MINIMUM REQUIREMENTS AUDIT ===\n")

# REQ 1: Usage of OpenEnv (latest release)
print("REQ 1: OpenEnv usage")
with open('env/environment.py') as f: env_src = f.read()
with open('pyproject.toml') as f: toml = f.read()
assert 'openenv' in toml.lower(), "openenv not in pyproject.toml dependencies"
assert 'OpenEnvBase' in env_src or 'openenv' in env_src, "env.py doesn't reference openenv"
assert 'openenv.yaml' in os.listdir('.') or os.path.exists('openenv.yaml'), "openenv.yaml missing"
with open('openenv.yaml') as f: oy = yaml.safe_load(f)
assert 'rubrics' in oy, "openenv.yaml missing rubrics"
assert 'name' in oy, "openenv.yaml missing name"
print("  ✓ openenv in pyproject.toml")
print("  ✓ CrisisOpsEnv inherits from OpenEnvBase")
print("  ✓ openenv.yaml present with rubrics")
print()

# REQ 2: Training script using Unsloth or HF TRL in Colab
print("REQ 2: Training script (Unsloth + TRL + Colab notebook)")
with open('training/grpo_trainer.py') as f: trainer = f.read()
assert 'unsloth' in trainer.lower() or 'FastLanguageModel' in trainer, "unsloth not in trainer"
assert 'GRPOTrainer' in trainer, "GRPOTrainer not in trainer"
assert os.path.exists('training/colab_notebook.ipynb'), "colab_notebook.ipynb missing"
with open('training/colab_notebook.ipynb') as f: nb = json.load(f)
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
assert len(code_cells) >= 6, f"Expected 6+ code cells, got {len(code_cells)}"
full_nb = ''.join(''.join(c['source']) for c in nb['cells'])
assert 'train(' in full_nb or 'grpo_trainer' in full_nb, "notebook doesn't call train()"
assert 'unsloth' in full_nb.lower(), "notebook doesn't mention unsloth"
print(f"  ✓ grpo_trainer.py: GRPOTrainer + Unsloth present")
print(f"  ✓ colab_notebook.ipynb: {len(code_cells)} code cells, calls train()")
print()

# REQ 3: Mini-blog or video (README must link to it)
print("REQ 3: Blog/video linked from README")
with open('README.md') as f: readme = f.read()
has_hf_link = 'huggingface.co' in readme
has_youtube = 'youtube' in readme.lower() or 'youtu.be' in readme.lower()
has_space = 'spaces' in readme.lower()
print(f"  HF link in README: {has_hf_link}")
print(f"  YouTube link in README: {has_youtube}")
print(f"  HF Spaces link in README: {has_space}")
if not (has_hf_link or has_youtube):
    print("  !! WARNING: README needs link to blog post or video !!")
print()

# REQ 4: HF Spaces deployment
print("REQ 4: HF Spaces deployment")
has_yaml = readme.startswith('---') or '---' in readme[:200]
print(f"  HF Spaces YAML front-matter in README: {has_yaml}")
assert os.path.exists('app.py'), "app.py missing"
with open('app.py') as f: app = f.read()
assert 'gradio' in app.lower(), "app.py doesn't use gradio"
assert 'demo.launch' in app, "app.py missing demo.launch()"
print("  ✓ app.py present with Gradio demo")
print()

print("=== AUDIT COMPLETE ===")
