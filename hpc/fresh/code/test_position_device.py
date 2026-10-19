"""Device wrapper regression checks. Author: James Edward Ball."""
import unittest
from types import SimpleNamespace
import torch
from extract import device_safe_position_encoding

class PositionChecks(unittest.TestCase):
    def test_cpu_values_and_shape_unchanged(self):
        values=torch.arange(24,dtype=torch.float32).reshape(1,3,8)
        encoder=SimpleNamespace(pos_embed=torch.zeros(1,3,8), interpolate_pos_encoding=lambda shape:values)
        device_safe_position_encoding(encoder)
        self.assertTrue(torch.equal(encoder.interpolate_pos_encoding((1,2,2)),values))
    def test_generated_cpu_tensor_follows_buffer_device(self):
        encoder=SimpleNamespace(pos_embed=torch.empty((1,3,8),device='meta',dtype=torch.float64), interpolate_pos_encoding=lambda shape:torch.ones(1,3,8))
        device_safe_position_encoding(encoder)
        value=encoder.interpolate_pos_encoding((1,2,2))
        self.assertEqual(value.device.type,'meta');self.assertEqual(value.dtype,torch.float64)
        self.assertEqual(tuple(value.shape),(1,3,8))

if __name__=='__main__':unittest.main()
