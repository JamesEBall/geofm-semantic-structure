"""Run isolated test modules without cross-study import collisions. James Edward Ball."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(root / 'tests')], check=True)
subprocess.run([sys.executable, str(root / 'global/support/test_support.py')], check=True)
for name in ['fresh', 'global', 'mechanistic', 'internal-layers', 'supplement-controls', 'supplement-uncertainty', 'supplement-regions']:
    subprocess.run([sys.executable, '-m', 'pytest', '-q', str(root / name / 'code')], check=True)
