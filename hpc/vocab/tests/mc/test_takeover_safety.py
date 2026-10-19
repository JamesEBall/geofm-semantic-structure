"""Mission Control safety checks. No network, scheduler commands, GPUs or real data.

Run: python -m unittest discover -s hpc/vocab/tests/mc -v
Author: James Edward Ball
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[2] / "code"
sys.path.insert(0, str(CODE))
from mc import runner
from mc.context import Ctx
from mc.core import FINALIZE, Store, atomic_path, atomic_write_json, sha256_file, worker_id
from mc.registry import Stage
from mc.slurm import FakeSlurm, enucc_nodes, parse_squeue_json
from mc.work import WorkView


class StateSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = Store(self.root)
        self.cfg = {"root": str(self.root), "_profile": "selftest", "mc": {"unit_backoff_s": 10}}
        self.ctx = Ctx(self.cfg, CODE.parent, self.root, self.root, self.root / "shared", self.store)
        self.ctx.log = lambda *a, **kw: None
        self.old_stop = runner._STOP["flag"]
        runner._STOP["flag"] = False
        self.addCleanup(runner._STOP.__setitem__, "flag", self.old_stop)
        self.env = patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)

    def stage(self, run=None, **kw):
        return Stage("task", "P1", "cpu", (), lambda c: [{"id": "one"}],
                     run or (lambda c, u, k: {}), **kw)

    def execute(self, st, unit="one"):
        return runner.execute(self.cfg, CODE.parent, st.name, unit, stages={st.name: st}, ctx=self.ctx)

    def ready(self):
        self.store.write_units("task", [{"id": "one"}])
        self.assertTrue(self.store.claim("task", "one"))

    def test_claim_exclusive_and_wrong_owner_cannot_release(self):
        self.assertTrue(self.store.claim("task", "one", "A"))
        self.assertFalse(self.store.claim("task", "one", "B"))
        self.store.release("task", "one", "B")
        self.assertEqual(self.store.claim_owner("task", "one")["worker"], "A")
        self.store.release("task", "one", "A")
        self.assertTrue(self.store.claim("task", "one", "B"))

    def test_claim_sync_failure_leaves_unit_available(self):
        with patch("mc.core.os.fsync", side_effect=OSError("filesystem I/O error")):
            with self.assertRaises(OSError):
                self.store.claim("task", "one", "failed-job")
        self.assertFalse(self.store.claim_path("task", "one").exists())
        self.assertTrue(self.store.claim("task", "one", "retry-job"))
        self.assertEqual(self.store.claim_owner("task", "one")["worker"], "retry-job")

    def test_killed_claim_writer_leaves_no_partial_owner(self):
        code = ("import os,sys;from unittest.mock import patch;from mc.core import Store;"
                "p=patch('mc.core.json.dump',side_effect=lambda *a,**k:os._exit(9));p.start();"
                "Store(sys.argv[1]).claim('task','one','killed-job')")
        r = subprocess.run([sys.executable, "-c", code, str(self.root)],
                           env={**os.environ, "PYTHONPATH": str(CODE)}, capture_output=True)
        self.assertEqual(r.returncode, 9, r.stderr)
        self.assertFalse(self.store.claim_path("task", "one").exists())
        self.assertEqual(self.store.claimed("task"), set())
        self.assertTrue(self.store.claim("task", "one", "retry-job"))

    def test_crashed_worker_claim_persists_until_explicit_recovery(self):
        # A killed worker cannot accidentally make its unit available to a second worker.
        code = ("import os,sys;from mc.core import Store;"
                "assert Store(sys.argv[1]).claim('task','one','dead-job');os._exit(9)")
        r = subprocess.run([sys.executable, "-c", code, str(self.root)],
                           env={**os.environ, "PYTHONPATH": str(CODE)}, capture_output=True)
        self.assertEqual(r.returncode, 9, r.stderr)
        self.assertFalse(self.store.claim("task", "one", "new-job"))
        self.assertEqual(self.store.claim_owner("task", "one")["worker"], "dead-job")
        # This is a manual recovery primitive, not proof of automated recovery.
        self.store.release("task", "one", "dead-job")
        self.assertTrue(self.store.claim("task", "one", "new-job"))

    def test_atomic_output_failure_preserves_previous_complete_file(self):
        dest = self.root / "result.bin"
        dest.write_bytes(b"previous")
        with self.assertRaises(OSError):
            with atomic_path(dest) as tmp:
                tmp.write_bytes(b"partial")
                raise OSError("disk full")
        self.assertEqual(dest.read_bytes(), b"previous")
        self.assertFalse(list(self.root.glob(".*.tmp.*")))

    def test_done_marker_written_after_output_and_contains_hash(self):
        def run(c, u, k):
            self.assertFalse(c.store.is_done("task", "one"))
            path = c.root / "result.bin"
            with atomic_path(path) as tmp:
                tmp.write_bytes(b"complete")
            return {"outputs": [path]}
        self.ready()
        self.assertEqual(self.execute(self.stage(run)), "done")
        rec = self.store.done_records("task")[0]
        self.assertEqual(rec["outputs"]["result.bin"]["sha256"], sha256_file(self.root / "result.bin"))
        self.assertIsNone(self.store.claim_owner("task", "one"))
        self.assertFalse(self.store.ckpt_dir("task", "one").exists())

    def test_missing_declared_output_never_commits(self):
        self.ready()
        st = self.stage(lambda c, u, k: {"outputs": [c.root / "missing"]})
        self.assertEqual(self.execute(st), "failed")
        self.assertFalse(self.store.is_done("task", "one"))
        self.assertEqual(self.store.attempts("task", "one"), 1)
        self.assertIsNone(self.store.claim_owner("task", "one"))

    def test_sigusr1_checkpoint_yields_and_resumes_without_retry_charge(self):
        previous = {s: signal.getsignal(s) for s in (signal.SIGUSR1, signal.SIGTERM)}
        runner.install_stop_handlers()
        self.addCleanup(lambda: [signal.signal(s, h) for s, h in previous.items()])
        seen = []
        def run(c, u, k):
            p = k.load("progress", {"next": 0})
            seen.append(p["next"])
            if p["next"] == 0:
                os.kill(os.getpid(), signal.SIGUSR1)
                self.assertTrue(k.due())
                k.save("progress", {"next": 3})
                k.maybe_stop()
            path = c.root / "result.bin"
            path.write_bytes(b"resumed from 3")
            return {"outputs": [path]}
        self.ready()
        st = self.stage(run)
        self.assertEqual(self.execute(st), "yielded")
        self.assertFalse(self.store.is_done("task", "one"))
        self.assertEqual(self.store.attempts("task", "one"), 0)
        self.assertIsNone(self.store.claim_owner("task", "one"))
        runner._STOP["flag"] = False
        self.assertTrue(self.store.claim("task", "one"))
        self.assertEqual(self.execute(st), "done")
        self.assertEqual(seen, [0, 3])

    def test_failure_keeps_checkpoint_for_retry(self):
        def run(c, u, k):
            k.save("progress", {"next": 7})
            raise RuntimeError("transient failure")
        self.ready()
        self.assertEqual(self.execute(self.stage(run)), "failed")
        path = self.store.ckpt_dir("task", "one") / "progress.json"
        self.assertEqual(json.loads(path.read_text()), {"next": 7})

    def test_pause_and_gate_block_workview(self):
        st = self.stage(gate="APPROVE_P2")
        view = WorkView(self.store, {"task": st}, self.cfg)
        self.assertEqual(view.state("task"), "gated")
        self.assertEqual(view.all_items()[0], [])
        self.store.set_flag("APPROVE_P2")
        self.assertEqual(view.state("task"), "ready")
        self.store.set_flag("PAUSE_DISK")
        self.assertEqual(view.all_items()[0], [])

    def test_direct_execute_cannot_bypass_science_gate(self):
        self.ready()
        called = []
        st = self.stage(lambda *a: called.append(True), gate="APPROVE_P2")
        self.assertEqual(self.execute(st), "gated")
        self.assertEqual(called, [])
        self.assertFalse(self.store.is_done("task", "one"))

    def test_direct_execute_cannot_bypass_dependency(self):
        self.ready()
        st = self.stage()
        st.deps = ("upstream",)
        self.assertEqual(self.execute(st), "blocked")
        self.assertFalse(self.store.is_done("task", "one"))

    def test_execute_never_releases_foreign_claim(self):
        self.store.write_units("task", [{"id": "one"}])
        self.store.claim("task", "one", "foreign-worker")
        called = []
        st = self.stage(lambda *a: called.append(True))
        outcome = self.execute(st)
        self.assertNotEqual(outcome, "done")
        self.assertEqual(called, [])
        self.assertEqual(self.store.claim_owner("task", "one")["worker"], "foreign-worker")

    def test_unit_retries_back_off_and_exhaustion_blocks_stage(self):
        self.store.write_units("task", [{"id": "one"}])
        st = self.stage(max_attempts=2)
        view = WorkView(self.store, {"task": st}, self.cfg)
        self.store.record_failure("task", "one", "transient", "try again")
        last = self.store.failures("task")["one"][0]["t"]
        self.assertEqual(view.stage_items("task", now=last + 9)[0], [])
        self.assertEqual([i.unit for i in view.stage_items("task", now=last + 11)[0]], ["one"])
        self.store.record_failure("task", "one", "transient", "exhausted")
        items, _ = view.stage_items("task", now=last + 100)
        self.assertEqual([i.unit for i in items], [FINALIZE])
        self.store.claim("task", FINALIZE)
        self.assertEqual(self.execute(st, FINALIZE), "failed")
        self.assertEqual(self.store.stage_status("task")["status"], "failed")

    def test_reserved_control_unit_ids_are_rejected(self):
        for uid in (runner.PLAN, FINALIZE):
            with self.subTest(uid=uid), self.assertRaises(ValueError):
                self.store.write_units("task", [{"id": uid}])

    def test_full_profile_cannot_use_selftest_autoapproval(self):
        cfg = {**self.cfg, "_profile": "full"}
        with self.assertRaisesRegex(ValueError, "selftest"):
            runner.local_run(cfg, CODE.parent, approve=True)

    def test_empty_stage_registry_is_an_error(self):
        with patch.object(runner, "load_stages", return_value={}):
            with self.assertRaisesRegex(ValueError, "No stages"):
                runner.local_run(self.cfg, CODE.parent)

    def test_pause_blocks_direct_execution_and_preserves_other_claim(self):
        self.store.write_units("task", [{"id": "one"}])
        self.store.claim("task", "one", "foreign-worker")
        self.store.set_flag("PAUSE_DISK")
        self.assertEqual(self.execute(self.stage()), "blocked")
        self.assertEqual(self.store.claim_owner("task", "one")["worker"], "foreign-worker")

    def test_plan_failure_obeys_backoff(self):
        st = self.stage()
        self.store.record_failure("task", runner.PLAN, "broken", "planner failure")
        last = self.store.failures("task")[runner.PLAN][0]["t"]
        items, _ = WorkView(self.store, {"task": st}, self.cfg).stage_items("task", now=last + 1)
        self.assertEqual(items, [])

    def test_finalize_exhaustion_does_not_retry_forever(self):
        self.store.write_units("task", [])
        st = self.stage(max_attempts=2)
        for _ in range(2):
            self.store.record_failure("task", FINALIZE, "broken", "finalizer failure")
        items, _ = WorkView(self.store, {"task": st}, self.cfg).stage_items("task", now=1e12)
        self.assertEqual(items, [])


class SchedulerSafety(unittest.TestCase):
    def test_same_user_si_job_is_distinct_from_mc_job(self):
        fake = FakeSlurm(enucc_nodes(), user="40021461")
        si = fake.add_job("40021461", "gpu", 16, 32000, 1, name="si-72b")
        own = fake.add_job("40021461", "gpu", 16, 32000, 1,
                           comment="megaearth|full|gpu|worker")
        self.assertIsNone(si.mc_tag())
        self.assertEqual(own.mc_tag(), ("full", "gpu", "worker"))
        # Explicit targeted cancellation leaves SI intact. Ownership policy is still missing.
        fake.cancel(own.id)
        self.assertEqual([j.id for j in fake.snapshot().jobs], [si.id])

    def test_signal_targets_only_selected_worker_and_does_not_cancel_si(self):
        fake = FakeSlurm(enucc_nodes(), user="40021461")
        si = fake.add_job("40021461", "gpu", 16, 32000, 1, state="RUNNING", nodes=["gpu01"], name="si-72b")
        own = fake.add_job("40021461", "gpu", 16, 32000, 1, state="RUNNING", nodes=["gpu02"],
                           comment="megaearth|full|gpu|worker")
        fake.cancel(own.id, signal="USR1", batch=True)
        self.assertEqual(fake.signals, [(own.id, "USR1")])
        self.assertEqual({j.id for j in fake.snapshot().jobs}, {si.id, own.id})

    def test_foreign_resources_are_accounted_and_reserved_nodes_unusable(self):
        fake = FakeSlurm(enucc_nodes())
        fake.add_job("someone", "gpu", 60, 240000, 2, state="RUNNING", nodes=["gpu01"])
        snap = fake.snapshot()
        self.assertEqual(snap.nodes["gpu01"].free_gpus, 0)
        fake.nodes["gpu02"].state = "IDLE+PLANNED"
        fake.nodes["gpu03"].state = "IDLE+RESERVED"
        fake.nodes["gpu04"].state = "IDLE+DRAIN"
        self.assertTrue(all(not fake.snapshot().nodes[n].usable for n in ("gpu02", "gpu03", "gpu04")))

    def test_scheduler_json_retains_foreign_restrictions_and_gpu_demand(self):
        doc = {"jobs": [{"job_id": 99, "name": "si-72b", "user_name": "40021461",
                         "job_state": ["PENDING"], "state_reason": "Resources", "partition": "gpu",
                         "cpus": {"set": True, "number": 16}, "node_count": 1,
                         "memory_per_cpu": 2048, "tres_per_node": "gres/gpu:a100:2",
                         "required_nodes": "gpu[01-02]", "excluded_nodes": "gpu03"}]}
        job = parse_squeue_json(json.dumps(doc))[0]
        self.assertEqual(job.gpus_per_node, 2)
        self.assertEqual(job.mem_per_node_mb, 32768)
        self.assertEqual(job.req_nodes, ["gpu01", "gpu02"])
        self.assertEqual(job.exc_nodes, ["gpu03"])
        self.assertIsNone(job.mc_tag())


if __name__ == "__main__":
    unittest.main()
