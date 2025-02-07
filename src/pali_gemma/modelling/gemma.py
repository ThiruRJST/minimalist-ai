import math
import torch
import torch.nn as nn

from src.pali_gemma.configs.siglip_config import SigLipVisionConfig
from src.pali_gemma.modelling.siglip import SigLipVisionModel
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, List



class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = SigLipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size
        
        language_model = GemmaForCasualLM(config.text_config)
        self.language_model = language_model
        
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
    
    def tie_weights(self):
        self.language_model.tie_weights()