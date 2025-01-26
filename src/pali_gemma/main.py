from configs.siglip_config import SigLipVisionConfig
from modelling.siglip import SigLipVisionModel
from torchsummary import summary


siglip_model = SigLipVisionModel(SigLipVisionConfig())
summary(siglip_model, (3, 224, 224))
