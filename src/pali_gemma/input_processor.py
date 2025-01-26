import torch
import numpy as np

from PIL import Image

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]

class PaliGemmaProc:
    IMAGE_TOKEN = "<image>"
    
    def __init__(self, tokenizer, num_image_tokens:int, image_size:int):
        
        super().__init__()
        
        self.num_image_tokens = num_image_tokens
        self.image_size = image_size
        
        tokens_to_add = {"additional_special_tokens": self.IMAGE_TOKEN}
        tokenizer.add_special_tokens(tokens_to_add)
        
        EXTRA_TOKENS = [
            f"<loc{i:.04d}>"
            for i in range(1024)
        ]#Tokens used for object detection
        
        EXTRA_TOKENS += [
            f"<seg{i:03d}"
            for i in range(128)
        ]#Tokens used for segmentation
        
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        #We will add the BOS and EOS tokens
        tokenizer.add_bos_tokens = False
        tokenizer.add_eos_tokens = False
        
        self.tokenizer = tokenizer
        
