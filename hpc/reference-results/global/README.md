# Global reference results

Author: James Edward Ball

The score arrays, geographic splits and expected estimates in this directory support numerical verification of the global experiment. Run from the repository root:

```sh
python hpc/check_global_results.py
python hpc/check_readout_references.py
python hpc/plot_global_results.py --out output/figures
```

The first command checks all 768 reported estimates against the 80 saved fold-score arrays and their hashes. The second checks the readout reference controls. Figure generation uses the saved estimates without fitting models.
