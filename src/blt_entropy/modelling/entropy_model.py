import os
import sys

from src.blt_entropy.configs import BLTEntropyModelConfig
from src.blt_entropy.modelling.base_transformer import BaseTransformer

def create_blt_entropy_model(config: BLTEntropyModelConfig):
    
    print(f"Creating BLTEntropyModel with config: {config}")
    return BaseTransformer(
        BLTEntropyModelConfig()
    )

if __name__ == "__main__":
    model = create_blt_entropy_model(BLTEntropyModelConfig())
    print(model)