"""Run checks in separate processes. Author: James Edward Ball."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--megaearth', action='store_true', help='Include MegaEarth and geometry calibration tests')
    a = p.parse_args()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT/'src'), str(ROOT/'hpc/vocab/code')]),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    def run(args):
        subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True)
    run(['-m', 'pytest', '-q', 'tests'])
    run(['hpc/test_all.py'])
    if a.megaearth:
        for group in ['tests', 'retained-cohort-correction-20261018',
                      'supplement-geometry-support-20261018',
                      'supplement-geometry-gram-pilot-20261018']:
            run(['-m', 'pytest', '-q', 'hpc/vocab/' + group])


if __name__ == '__main__':
    main()
