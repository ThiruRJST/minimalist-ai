import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.blt_entropy.configs import BLTEntropyModelConfig
from src.blt_entropy.modelling.base_transformer import BaseTransformer

def test_blt_entropy_model():
    
    print(f"Creating BLTEntropyModel with config: {BLTEntropyModelConfig()}")
    blt_entropy_model =  BaseTransformer(
        BLTEntropyModelConfig()
    )
    
    assert blt_entropy_model != None
    assert type(blt_entropy_model) == BaseTransformer