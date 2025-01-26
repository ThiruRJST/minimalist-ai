import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pali_gemma.configs.siglip_config import SigLipVisionConfig
from src.pali_gemma.modelling.siglip import SigLipVisionModel

def test_pali_gemma_load():
    siglip_model = SigLipVisionModel(SigLipVisionConfig())
    assert siglip_model != None
    input_x = torch.randn(1, 3, 224, 224)
    output = siglip_model(input_x)
    assert output != None