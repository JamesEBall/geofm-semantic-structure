import unittest
import numpy as np
from uncertainty import draws,paired_intervals
class Checks(unittest.TestCase):
 def test_same_draw_and_complete_blocks(self):
  blocks=np.repeat(['a','b','c'],[2,3,4]);a=draws(blocks,2);np.testing.assert_array_equal(a,draws(blocks,2))
  for block in set(blocks):self.assertEqual(np.sum(blocks[a]==block)%np.sum(blocks==block),0)
 def test_no_validation_or_test_in_draw(self):
  split=np.array(['train']*9+['val']*3+['test']*4);train=np.flatnonzero(split=='train');blocks=np.repeat(['a','b','c'],3)
  self.assertTrue(np.all(split[train[draws(blocks,1)]]=='train'))
 def test_identical_condition_zero(self):
  values=np.array([[0,0,np.nan,np.nan],[np.nan,np.nan,0,0.]])
  result=paired_intervals(values,np.array(['a','b','a','b']),100)
  self.assertEqual(result['estimate'],0);self.assertEqual(result['conditional_95_interval'],[0,0])
 def test_category_and_block_weighting(self):
  values=np.array([[1,1,1,3,np.nan,np.nan],[np.nan,np.nan,np.nan,np.nan,4,4]])
  result=paired_intervals(values,np.array(['a','a','a','b','a','b']),100)
  self.assertEqual(result['estimate'],3.)
if __name__=='__main__':unittest.main()
