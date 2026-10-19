"""Batch requested F2 paths; retain the original verifier and every shard check.

Author: James Edward Ball. No verification result or path lookup is cached.
"""


def verify_raw_inputs(runner, ctx, unit):
    if ctx.stage not in ('features_fit', 'features', 'extract'):
        return
    import pandas as pd
    shards = []

    def required_paths():
        # The unchanged verifier hashes each yield before requesting the next.
        # Both tables are therefore verified before the table is read for shards.
        yield ctx.p('assemble/windows_qa.parquet')
        yield ctx.p('admit/windows_final.parquet')
        if ctx.stage == 'extract':
            table = pd.read_parquet(ctx.p('assemble/windows_qa.parquet'))
            shards.extend(table.sort_values('widx').iloc[unit['start']:unit['stop']].shard_dir.unique())
            yield ctx.shared / 'weights/dofa/DOFA_ViT_base_e100.pth'
        else:
            shards.append(unit['shard_dir'])
        for shard in shards:
            yield ctx.run / shard / 'manifest.json'

    # Same verifier, same complete input set and order, one F2 path traversal.
    runner.verify_frozen_paths(ctx, required_paths())
    if ctx.stage == 'extract':
        from stages._imagery_io import verify_shard
        for shard in shards:
            verify_shard(ctx.run / shard)
