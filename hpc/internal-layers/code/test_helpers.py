import unittest
import numpy as np
import torch
from run import attach, perturb, select

class Helpers(unittest.TestCase):
    def test_energy_sham_cls(self):
        x=torch.randn(3,5,8);d=np.eye(8,dtype=np.float32)[0];r=np.eye(8,dtype=np.float32)[1]
        self.assertIs(perturb(x,d,0),x)
        target=perturb(x,d,.5);random=perturb(x,d,.5,r)
        norm=perturb(x,d,.5,'norm-only')
        torch.testing.assert_close(torch.linalg.vector_norm(target,dim=-1),torch.linalg.vector_norm(norm,dim=-1))
        torch.testing.assert_close(torch.nn.functional.normalize(norm[:,1:],dim=-1),torch.nn.functional.normalize(x[:,1:],dim=-1))
        torch.testing.assert_close(target[:,0],x[:,0],rtol=0,atol=0)
        torch.testing.assert_close(torch.linalg.vector_norm(target-x,dim=-1),torch.linalg.vector_norm(random-x,dim=-1))
        torch.testing.assert_close(perturb(x,d,1)[:,1:,0],torch.zeros_like(x[:,1:,0]))
    def test_post_residual_hook(self):
        class Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__();self.layers=torch.nn.ModuleList([torch.nn.ModuleList([torch.nn.Linear(4,4),torch.nn.Linear(4,4)]) for _ in range(4)]);self.norm=torch.nn.LayerNorm(4)
            def forward(self,x):
                for attn,ff in self.layers:x=attn(x)+x;x=ff(x)+x
                return self.norm(x)
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.transformer=Transformer()
        m=Model();x=torch.randn(2,5,4);clean=m.transformer(x);seen=[]
        _,detach=attach(m,'clay',lambda i,t:(seen.append(i) or t))
        torch.testing.assert_close(clean,m.transformer(x),rtol=0,atol=0);self.assertEqual(seen,[1,2,3,4]);detach()
    def test_selection(self):
        rows=[dict(split=s,label=c,block=b,id=i) for s in ['train','val','test'] for c in ['a','b'] for b in range(5) for i in range(9)]
        ids=select(rows,60);self.assertEqual(len(ids),60);self.assertEqual(len(set(ids)),60)
        for s in ['train','val','test']:
            for c in ['a','b']:self.assertEqual(len({rows[i]['block'] for i in ids if rows[i]['split']==s and rows[i]['label']==c}),5)
        np.testing.assert_array_equal(ids,select(rows,60))

if __name__=='__main__':unittest.main()
