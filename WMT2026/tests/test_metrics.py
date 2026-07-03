import pytests
import torch
import torch.distributions as td



def test_entropy():
    '''
    tests entropy computation per batch
    '''
    data = torch.tensor([[1.0, 2.0, 3.0], [3.0,4.0,5.0]])
    data_entropy = td.Categorical(data)
    data_entropy = data_entropy.entropy()
    assert data_entropy == torch.tensor([[], []])