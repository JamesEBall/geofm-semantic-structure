import unittest
from pathlib import Path
import numpy as np
import controls as m
class Checks(unittest.TestCase):
 def test_descriptor_original(self):
  rng=np.random.default_rng(3);img=rng.uniform(.01,.8,(6,64,64));names,x,_=m.descriptors(dict(zip(m.BANDS,img)),{b:np.ones((64,64),bool) for b in m.BANDS});self.assertEqual(len(names),116);self.assertTrue(np.isfinite(x).all())
 def test_shuffle_preserves_blocks(self):
  y=np.array([0,1,2,2,2,2]);blocks=np.array([0,0,0,1,1,1]);s=m.shuffle_within_blocks(y,blocks,101)
  for b in (0,1):np.testing.assert_array_equal(np.sort(s[blocks==b]),np.sort(y[blocks==b]))
 def test_nearest_exact(self):
  train=np.array([[0.,0.],[1.,1.],[2.,2.]]);query=np.array([[.2,.2],[1.9,1.9]])
  found=m.neighbours(train,query,np.array([0,1,2]),3,1)[:,0].argmax(1);np.testing.assert_array_equal(found,[0,2])
 def test_metadata_no_dates(self):
  x,n=m.metadata([{'lon':0,'lat':0}]);self.assertEqual(x.shape,(1,3));np.testing.assert_array_equal(x,[[1,0,0]])
 def test_fold_fresh_and_global(self):
  for study in ('fresh','global'):
   api=m.load_api(Path(__file__).parents[2]/study);c=len(api.CLASSES);rng=np.random.default_rng(12);y=np.tile(np.arange(c),9);splits=np.repeat(['train','val','test'],3*c);blocks=np.array([str(i//c) for i in range(9*c)]);x=rng.normal(size=(9*c,12));text={r:rng.normal(size=(c,6)) for r in api.REFS}
   result,cache=m.evaluate_fold(x,y,splits,blocks,text,api,(0,1));self.assertEqual(result['support']['test'],6);self.assertEqual(len([k for k in result if '/' in k]),24);self.assertTrue(all(np.isfinite(v).all() for k,v in cache.items() if '__' in k))
if __name__=='__main__':unittest.main()
