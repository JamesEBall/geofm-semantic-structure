"""Effective-code provenance and fail-closed input tests. Author: James Edward Ball."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

HERE=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

worker=load('amended_worker_test',HERE/'amended_worker.py')


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.base=self.root/'base';code=self.base/'retained-cohort-correction-20261018';code.mkdir(parents=True)
        (code/'run.py').write_text('base run');(code/'controller.py').write_text('base controller')
        self.overlay=self.root/'overlay';self.overlay.mkdir()
        names=['batched_f2.py','amended_worker.py','worker_amended.sbatch','controller_resume.py']
        for name in names:(self.overlay/name).write_text(name)
        self.output=self.root/'output';self.output.mkdir()
        self.ctx=SimpleNamespace(run=self.output,p=lambda *v:self.output.joinpath(*v),cfg={'analysis_input_freeze':'f3_cpu'})
        for name in ('_retained/identity.json','freeze_f2/manifest.json'):
            path=self.ctx.p(name);path.parent.mkdir(parents=True);path.write_text('{}')
        self.doc={'amendment_id':'test','base_release':str(self.base),'base_runner_sha256':worker.sha(code/'run.py'),
                  'base_controller_sha256':worker.sha(code/'controller.py'),
                  'base_scientific_identity_sha256':worker.sha(self.ctx.p('_retained/identity.json')),
                  'base_f2_sha256':worker.sha(self.ctx.p('freeze_f2/manifest.json')),
                  'files':{name:worker.sha(self.overlay/name) for name in names}}
        self.manifest=self.overlay/'manifest.json';self.manifest.write_text(json.dumps(self.doc))
        self.runner=load('base_runner_worker_test',HERE.parent/'run.py')

    def tearDown(self):self.temp.cleanup()

    def test_manifest_pins_base_and_all_effective_overlay_code(self):
        worker.validate_manifest(self.manifest,self.base)
        (self.overlay/'batched_f2.py').write_text('modified')
        with self.assertRaisesRegex(ValueError,'Overlay file hash changed'):worker.validate_manifest(self.manifest,self.base)

    def test_manifest_rejects_base_mutation_and_inventory_escape(self):
        path=self.base/'retained-cohort-correction-20261018/run.py';path.write_text('changed')
        with self.assertRaisesRegex(ValueError,'Original runner'):worker.validate_manifest(self.manifest,self.base)
        path.write_text('base run');self.doc['files']['../outside']='0'*64;self.manifest.write_text(json.dumps(self.doc))
        with self.assertRaisesRegex(ValueError,'escapes amendment'):worker.validate_manifest(self.manifest,self.base)

    def test_every_receipt_has_execution_provenance_and_same_output_hash(self):
        helper=SimpleNamespace(verify_raw_inputs=Mock())
        execution,check=worker.install(self.runner,helper,self.manifest,self.doc,self.output)
        path=self.ctx.p('features/a.npy');path.parent.mkdir();path.write_bytes(b'unchanged output')
        self.runner.record(self.ctx,'features','a',{'id':'a'},[path])
        rec=self.runner.read(self.runner.receipt_path(self.ctx,'features','a'))
        self.assertEqual(rec['execution'],execution)
        self.assertEqual(rec['identity'],self.doc['base_scientific_identity_sha256'])
        self.assertEqual(rec['outputs'],{'features/a.npy':worker.sha(path)})
        self.runner.validate_receipt(self.ctx,'features','a')
        foreign=self.root/'foreign';foreign.write_text('x')
        with self.assertRaisesRegex(ValueError,'outside the derivative'):
            self.runner.record(self.ctx,'features','b',{'id':'b'},[foreign])

    def test_manifest_and_identity_rechecked_before_each_raw_call(self):
        helper=SimpleNamespace(verify_raw_inputs=Mock())
        worker.install(self.runner,helper,self.manifest,self.doc,self.output)
        self.runner.verify_raw_inputs(self.ctx,{'id':'a'})
        for name in ('freeze_f2/manifest.json','_retained/identity.json'):
            p=self.ctx.p(name);before=p.read_text();p.write_text('{"changed":true}')
            with self.assertRaises(ValueError):self.runner.verify_raw_inputs(self.ctx,{'id':'a'})
            p.write_text(before)
        self.assertEqual(helper.verify_raw_inputs.call_count,1)


if __name__=='__main__':unittest.main()
