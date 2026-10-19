"""Numerical and split-isolation checks. Author: James Edward Ball."""
import unittest
from pathlib import Path
import numpy as np
import intervene as m

class Checks(unittest.TestCase):
    def test_projection(self):
        rng=np.random.default_rng(9); x=rng.normal(size=(30,8)); q=rng.normal(size=8); q/=np.linalg.norm(q)
        np.testing.assert_allclose(m.erase(x,q)@q,0,atol=1e-14)
        np.testing.assert_array_equal(m.erase(x,q,0),x)
    def test_refit_against_direct(self):
        rng=np.random.default_rng(4); x=rng.normal(size=(35,8)); x-=x.mean(0); y=rng.normal(size=(35,3)); y-=y.mean(0)
        q=rng.normal(size=8); q/=np.linalg.norm(q); alpha=10.
        gram=x.T@x+alpha*np.eye(8); w=np.linalg.solve(gram,x.T@y)
        got=m.refit_weights(w,np.linalg.solve(gram,q),q); xp=m.erase(x,q)
        expected=np.linalg.solve(xp.T@xp+alpha*np.eye(8),xp.T@y)
        np.testing.assert_allclose(got,expected,atol=1e-13)
    def test_energy(self):
        rng=np.random.default_rng(7); x=rng.normal(size=(50,8))*np.arange(1,9); q=np.eye(8)[0]; r=np.eye(8)[7]
        sq,sr,eq,er=m.energy_strengths(x,q,r)
        self.assertLessEqual(max(sq,sr),1.)
        self.assertAlmostEqual(np.mean((x-m.erase(x,q,sq))**2),np.mean((x-m.erase(x,r,sr))**2))
    def test_synthetic_e2e_and_test_isolation(self):
        api=m.load_analysis(Path(__file__).parents[2]/'fresh/code/analyse.py')
        rng=np.random.default_rng(1); y=np.tile(np.arange(10),9); splits=np.repeat(['train','val','test'],30); blocks=np.array([str(i//10) for i in range(90)])
        x=rng.normal(size=(90,12))+np.eye(12)[y]; text={r:rng.normal(size=(10,8)) for r in api.REFS}
        scores={}; out=m.run_model(x,y,splits,blocks,text,api,[13],20,scores)
        changed=x.copy(); changed[splits=='test']*=100
        other=m.run_model(changed,y,splits,blocks,text,api,[13],20)
        self.assertEqual(out['selection'],other['selection'])
        self.assertEqual(len(out['categories']),10)
        self.assertEqual(len(scores),330)
        for c in api.CLASSES:
            self.assertEqual(out['categories'][c]['training_target_energy'],other['categories'][c]['training_target_energy'])
        self.assertTrue(all(np.isfinite(v).all() for v in scores.values()))

if __name__=='__main__': unittest.main()
