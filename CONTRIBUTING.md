# Contributing

Author: James Edward Ball

Use an issue to describe a reproducibility failure, with the commit, command, environment and relevant error output. Remove credentials and private paths before posting logs. Proposed fixes can be submitted as pull requests.

Keep frozen inputs, split identifiers, checksums and result receipts unchanged. Put new experiments and corrected-data analyses in a separately identified directory. Report the original result, the changed assumption and the new result together. Preserve failed calibration checks and adverse sensitivity results.

Use Python 3.12 and the separate test environment in [the reproduction guide](docs/REPRODUCIBILITY.md). Run tests relevant to changed behaviour. Explain any numerical tolerance and report the checks run in the pull request.

Contributions to original study code use the MIT licence. Preserve upstream licence files and author attribution for third-party code. Dataset and checkpoint terms remain separate.
