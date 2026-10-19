"""Split isolation and missing-support tests. James Edward Ball."""
import unittest
import numpy as np
import run
class ProtocolTests(unittest.TestCase):
 def setUp(self):
  self.rows=[dict(id=f'{g}-{s}-{c}-{b}',region=g,split=s,label=c,block=f'{g}-{s}-{b}') for g in run.GROUPS for s in ['train','val','test'] for c in run.core.CLASSES for b in range(2)]
 def test_complete_group_exclusion(self):
  for g in run.GROUPS:
   for held in [None]+run.core.FOLDS:
    f,v,t=run.split_masks(self.rows,g,held)
    self.assertFalse((f&v|f&t|v&t).any())
    for i,r in enumerate(self.rows):
     if f[i] or v[i]:self.assertNotEqual(r['region'],g)
     if t[i]:self.assertEqual(r['region'],g)
     if held is not None and (f[i] or v[i]):self.assertNotIn(run.core.CLASSES.index(r['label']),held)
    self.assertTrue(set(r['split'] for i,r in enumerate(self.rows) if t[i])=={'train','val','test'})
 def test_missing_test_category_reported(self):
  rows=[r for r in self.rows if not(r['region']=='Africa' and r['label']=='Water')];rng=np.random.default_rng(19);x=rng.normal(size=(len(rows),5));text={ref:rng.normal(size=(8,7)) for ref in run.core.REFS}
  d,_=run.evaluate(rows,x,text,'Africa');self.assertEqual(d['folds']['known']['status'],'unsupported');self.assertEqual(d['folds']['joint_0']['status'],'unsupported');self.assertTrue(all(v is None for v in d['joint_summary'].values()));self.assertEqual(d['folds']['joint_1']['status'],'complete')
if __name__=='__main__':unittest.main()
