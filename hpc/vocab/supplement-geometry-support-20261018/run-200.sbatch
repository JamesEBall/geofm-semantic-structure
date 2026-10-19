#!/bin/bash
#SBATCH --job-name=geometry-confirm
#SBATCH --partition=himem
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --nice=1000
set -euo pipefail
# Operational chunk amendment only; GEOMETRY_RELEASE keeps frozen scientific source.
: "${GEOMETRY_RELEASE:?}" "${GEOMETRY_RUN:?}" "${GEOMETRY_MODE:?}"
export VOCAB_ROOT=/mnt/scratch/users/40021461/geofm-semantic-expansion/vocab-01
export VOCAB_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$VOCAB_ROOT/env/activate.sh"
cd "$GEOMETRY_RELEASE"
if [[ "$GEOMETRY_MODE" == pilot ]]; then
  python -m pytest -q -p no:cacheprovider test_calibration.py
  python calibration.py truth --output "$GEOMETRY_RUN/population-truth.json"
  python calibration.py run --phase development --pilot --workers "$SLURM_CPUS_PER_TASK" --output "$GEOMETRY_RUN"
else
  python - "$GEOMETRY_MODE" "$SLURM_ARRAY_TASK_ID" "$GEOMETRY_RUN" "$SLURM_CPUS_PER_TASK" <<'PY'
import json,subprocess,sys
phase,task,out,workers=sys.argv[1:]
cfg=json.load(open('CONFIG.json'))
chunk=16 if phase=='development' else 200
units=[]
for i,s in enumerate(cfg['scenarios']):
 n=cfg['confirmation_null_replicates'] if phase=='confirmation' and s['beta']==0 else cfg[phase+'_replicates']
 units.extend((i,start,min(chunk,n-start)) for start in range(0,n,chunk))
i,start,count=units[int(task)]
subprocess.run([sys.executable,'calibration.py','run','--phase',phase,'--scenario',str(i),'--start',str(start),'--count',str(count),'--workers',workers,'--output',out],check=True)
PY
fi
