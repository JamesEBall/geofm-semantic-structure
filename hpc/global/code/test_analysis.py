"""Eight-class transfer and leakage guards. Author: James Edward Ball."""
import unittest
import numpy as np
import analyse as a

class GlobalAnalysis(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(42)
        y=np.tile(np.repeat(np.arange(a.N_CLASSES),3),3)
        splits=np.repeat(['train','val','test'],a.N_CLASSES*3)
        blocks=np.array([f'{s}-{i%3}' for i,s in enumerate(splits)])
        x=np.eye(a.N_CLASSES)[y]+rng.normal(0,.05,(len(y),a.N_CLASSES))
        t={r:rng.normal(size=(a.N_CLASSES,12)) for r in a.REFS}
        return x,y,splits,blocks,t
    def test_eight_class_geometry_and_transfer(self):
        x,y,s,b,t=self.fixture();result,cache=a.evaluate_model(x,y,s,b,t)
        for ref in a.REFS:
            self.assertEqual(len(result['geometry'][ref]['image_distances']),28)
            self.assertEqual(len(result['heldout'][ref]),4)
            self.assertEqual(len(cache['known'][ref][0][1][0]),8)
            self.assertEqual(len(result['known'][ref]['class_ap']),8)
            for fold,pair in zip(result['heldout'][ref],a.FOLDS):
                self.assertEqual(fold['n_train'],18)
                self.assertEqual(fold['supported_classes'],[a.CLASSES[c] for c in pair])
    def test_paired_identical_bootstrap_zero(self):
        x,y,s,b,t=self.fixture();_,cache=a.evaluate_model(x,y,s,b,t)
        result=a.paired_bootstrap({'appearance':cache,'same':cache},b[s=='test'],100,9)
        for track in ['known','heldout']:
            for ref in a.REFS:self.assertEqual(result['same'][track][ref]['conditional_query_ap_difference_interval'],[0.,0.])
    def test_reject_cross_split_block(self):
        rows=[dict(id=f'{s}-{c}',label=c,split=s,block=s,lon=10,lat=20) for s in ['train','val','test'] for c in a.CLASSES]
        a.validate_manifest(rows);rows[0]['block']='test'
        with self.assertRaisesRegex(ValueError,'crosses splits'):a.validate_manifest(rows)
    def test_ridge_fitting_independent_of_test(self):
        x,y,s,b,t=self.fixture()
        first=list(a.ridge_predictions(x[s=='train'],np.eye(8)[y[s=='train']],x[s=='val'],x[s=='test']))
        second=list(a.ridge_predictions(x[s=='train'],np.eye(8)[y[s=='train']],x[s=='val'],x[s=='test']*100))
        for one,two in zip(first,second):np.testing.assert_array_equal(one[1],two[1])

if __name__=='__main__':unittest.main()
