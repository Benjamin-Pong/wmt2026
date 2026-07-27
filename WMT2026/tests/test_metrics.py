import pytest
import torch
import torch.distributions as td

'''
Tests functionalities from metrics.py
'''

@pytest.fixture
def load_results():

    pass
'''
def test_entropy():
    #tests computation
    data = torch.tensor([[1.0, 2.0, 3.0], [3.0,4.0,5.0]])
    data_entropy = td.Categorical(data)
    data_entropy = data_entropy.entropy()
    assert data_entropy == torch.tensor([[], []])

def test_subword_probs_extraction():
    
    This function tests that the dimensions of subword_probs, logits, tokens, token ids are the same
    
    assert len(subword_probs)==len(tokens)==len(token_ids)==len(logits)
'''






