import pytest
import torch
from distributionshift import DistributionShift, create_distributionshift

class TestDistributionShift:
    def test_init(self):
        m = DistributionShift(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = DistributionShift(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_distributionshift()
        assert isinstance(m, DistributionShift)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
