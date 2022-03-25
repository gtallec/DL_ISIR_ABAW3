from models.encoders.inception import Inceptionv3, Inceptionv3Attention
from models.encoders.vision_transformer import VisionTransformer

SUPPORTED_ENCODERS = {"inceptionv3_encoder": Inceptionv3,
                      "iv3_att": Inceptionv3Attention,
                      "ViT": VisionTransformer}
