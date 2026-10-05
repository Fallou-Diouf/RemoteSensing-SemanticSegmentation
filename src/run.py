import subprocess
import sys

print("\n" + "=" * 60)
print("STEP 1 — TRAINING")
print("=" * 60)

subprocess.run([sys.executable, "src/train.py"], check=True)

print("\n" + "=" * 60)
print("STEP 2 — EVALUATION")
print("=" * 60)

subprocess.run([sys.executable, "src/evaluate.py"], check=True)

print("\n" + "=" * 60)
print("STEP 3 — INFERENCE")
print("=" * 60)

subprocess.run([sys.executable, "src/inference.py"], check=True)

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED")
print("=" * 60)