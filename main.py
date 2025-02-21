from src import logger
from src.blt_entropy.configs import BLTEntropyModelConfig
from src.blt_entropy.modelling.base_transformer import BaseTransformer

from torchsummary import summary


def create_blt_entropy_model(config: BLTEntropyModelConfig) -> BaseTransformer:
    logger.info(f"Creating BLT Entropy Model with config: {config}")
    model = BaseTransformer(
        config
    )
    return model

if __name__ == "__main__":
    config = BLTEntropyModelConfig()
    model = create_blt_entropy_model(config)
    logger.info(f"BLT Entropy Model created: {model}")
    
    summary(model, (72, 768))