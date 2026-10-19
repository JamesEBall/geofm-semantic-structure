"""~🌍 MegaEarth Mission Control 🛰️~ — elastic, checkpointed work orchestration on Slurm.

Stdlib only, so the governor can run under the system python3.11 as well as the project venv.

Model:
  * the pipeline is a DAG of *stages* (code/stages/registry.py);
  * each stage is split into small idempotent *units* (minutes each); a unit is complete only
    when its done-marker exists (the commit point, written last, with output hashes);
  * elastic *workers* (Slurm jobs) pull units from a lane, heartbeat, honour yield/terminate
    signals and checkpoint mid-unit when a unit supports it;
  * one *governor* job decides how many workers each lane gets: greedy when the cluster has
    no foreign demand, shrinking back to a small floor as soon as someone else is waiting.
"""

__version__ = "1.0.0"
