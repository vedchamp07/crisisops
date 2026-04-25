"""
main.py — CrisisOps v2 entry points.

Usage:
    python main.py demo          # launch Gradio demo (same as python app.py)
    python main.py calibrate     # run environment calibration
    python main.py train         # run GRPO training (requires GPU + training deps)
    python main.py eval          # run baseline evaluation
"""

import sys

def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if cmd == "demo":
        import app  # noqa: F401  — Gradio launches on import
    elif cmd == "calibrate":
        from calibration.calibrate import run_calibration
        result = run_calibration(n_episodes=20, seed=42)
        print(f"Status: {result['status']}")
        print(f"Greedy: {result['greedy_mean']:.3f}  Oracle: {result['oracle_mean']:.3f}  Gap: {result['gap']:.3f}")
    elif cmd == "train":
        from training.grpo_trainer import train
        train()
    elif cmd == "eval":
        import subprocess
        subprocess.run([sys.executable, "-m", "baselines.llm_agent", "--episodes", "10"])
    else:
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    main()
