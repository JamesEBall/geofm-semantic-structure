"""Batching integrity and parity tests. Author: James Edward Ball."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import pandas as pd

HERE=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

batch=load('batched_f2',HERE/'batched_f2.py')
runner=load('base_runner',HERE.parent/'run.py')


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.run=Path(self.tmp.name).resolve()
        self.ctx=SimpleNamespace(run=self.run,shared=self.run/'shared',stage='extract',p=lambda *v:self.run.joinpath(*v))
        self.unit={'start':0,'stop':4}
        self.table=pd.DataFrame({'widx':[0,1,2,3], 'shard_dir':['shard0','shard0','shard1','shard2']})
        self.paths=[self.ctx.p('assemble/windows_qa.parquet'),self.ctx.p('admit/windows_final.parquet'),self.ctx.shared/'weights/dofa/DOFA_ViT_base_e100.pth']
        self.paths += [self.run/f'shard{i}/manifest.json' for i in range(3)]
        for i,p in enumerate(self.paths):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(str(i))
        runner.write_once(self.ctx.p('freeze_f2/manifest.json'),{'files':{str(p):runner.digest(p) for p in self.paths}})
        self.parquet=patch.object(pd,'read_parquet',return_value=self.table);self.parquet.start()
        self.shards=patch('stages._imagery_io.verify_shard');self.shard_check=self.shards.start()

    def tearDown(self):
        self.shards.stop();self.parquet.stop();self.tmp.cleanup()

    def test_one_lookup_and_every_requested_hash_and_shard(self):
        with patch.object(runner,'read',wraps=runner.read) as read,patch.object(runner,'digest',wraps=runner.digest) as digest:
            batch.verify_raw_inputs(runner,self.ctx,self.unit)
            self.assertEqual(read.call_count,1)
            self.assertEqual([c.args[0] for c in digest.call_args_list],self.paths)
        self.assertEqual([c.args[0] for c in self.shard_check.call_args_list],[self.run/f'shard{i}' for i in range(3)])

    def test_original_and_batch_hash_identical_inputs_and_output(self):
        def run(verify):
            self.shard_check.reset_mock()
            with patch.object(runner,'digest',wraps=runner.digest) as digest:
                verify(self.ctx,self.unit)
                inputs=[(str(c.args[0]),runner.digest(c.args[0])) for c in list(digest.call_args_list)]
            return inputs,[c.args[0] for c in self.shard_check.call_args_list],sum(int(p.read_text()) for p in self.paths)
        original=run(runner.verify_raw_inputs)
        amended=run(lambda c,u:batch.verify_raw_inputs(runner,c,u))
        self.assertEqual(original,amended)

    def test_each_call_rehashes_every_input_and_rereads_f2(self):
        with patch.object(runner,'read',wraps=runner.read) as read,patch.object(runner,'digest',wraps=runner.digest) as digest:
            for _ in range(2):batch.verify_raw_inputs(runner,self.ctx,self.unit)
            self.assertEqual(read.call_count,2);self.assertEqual(digest.call_count,2*len(self.paths))

    def test_same_size_mtime_mutation_rejected_by_both(self):
        batch.verify_raw_inputs(runner,self.ctx,self.unit)
        p=self.paths[-1];s=p.stat();p.write_text('9');os.utime(p,ns=(s.st_atime_ns,s.st_mtime_ns))
        for verify in (runner.verify_raw_inputs,lambda c,u:batch.verify_raw_inputs(runner,c,u)):
            with self.assertRaisesRegex(ValueError,'changed since derivative F2'):verify(self.ctx,self.unit)

    def test_manifest_expected_hash_change_detected(self):
        batch.verify_raw_inputs(runner,self.ctx,self.unit)
        p=self.ctx.p('freeze_f2/manifest.json');d=runner.read(p);d['files'][str(self.paths[0])]='0'*64;p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'changed since derivative F2'):batch.verify_raw_inputs(runner,self.ctx,self.unit)

    def test_symlink_retarget_is_resolved_and_rejected(self):
        p=self.paths[-1];target=self.run/'target';p.rename(target);p.symlink_to(target)
        freeze=self.ctx.p('freeze_f2/manifest.json');d=runner.read(freeze);d['files'][str(target)]=d['files'].pop(str(p));freeze.write_text(json.dumps(d))
        batch.verify_raw_inputs(runner,self.ctx,self.unit)
        other=self.run/'other';other.write_text('changed');p.unlink();p.symlink_to(other)
        with self.assertRaisesRegex(ValueError,'changed since derivative F2'):batch.verify_raw_inputs(runner,self.ctx,self.unit)

    def test_cpu_stage_one_lookup_three_hashes(self):
        self.ctx.stage='features'
        with patch.object(runner,'read',wraps=runner.read) as read,patch.object(runner,'digest',wraps=runner.digest) as digest:
            batch.verify_raw_inputs(runner,self.ctx,{'shard_dir':'shard0'})
            self.assertEqual(read.call_count,1);self.assertEqual(digest.call_count,3)
        self.shard_check.assert_not_called()

    def test_tables_are_hashed_before_parquet_read(self):
        seen=[]
        real_digest=runner.digest
        def digest(path):
            seen.append(path)
            return real_digest(path)
        def read_table(path):
            self.assertEqual(seen,self.paths[:2])
            return self.table
        with patch.object(runner,'digest',side_effect=digest),patch.object(pd,'read_parquet',side_effect=read_table):
            batch.verify_raw_inputs(runner,self.ctx,self.unit)

    def test_nonraw_stage_no_io(self):
        self.ctx.stage='analysis_readouts'
        with patch.object(runner,'verify_frozen_paths') as verify:
            batch.verify_raw_inputs(runner,self.ctx,{})
            verify.assert_not_called()


if __name__=='__main__':unittest.main()
