# CUDA adapter amendment, 15 October 2026

Author: James Edward Ball

Job 174706 stopped at its numerical gate before extraction or test scoring. Clay and Prithvi 100M passed; Prithvi 300M raised a CPU/CUDA device mismatch. The pinned upstream interpolation regenerates positional embeddings from NumPy on CPU for the 64-pixel input. The adapter now moves the resulting positional tensor to the encoder buffer's device and dtype before use. Its values, preprocessing, checkpoint, input resolution, numerical tolerance and scientific protocol are unchanged. The wrapper applies to trained and random Prithvi 300M routes.

The provenance field said B08 although the frozen preparation manifest and actual source indices specify B8A. It now reports B8A. No image values or band choices changed. Clay still uses its pinned nir08 normalisation/wavelength configuration for the available narrow-NIR channel; this is an existing input-domain mismatch and is retained explicitly rather than silently changing preprocessing.

The failed parity report, logs, original adapter and original specification are preserved under amendments/2026-10-15-cuda-position. Freeze the corrected specification before replacement jobs. Keep the same four-scene gate and tolerances; resume extraction only after all checks and batch benchmarks pass. No test results informed this engineering amendment.
