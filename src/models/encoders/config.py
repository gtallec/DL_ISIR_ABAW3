from models.encoders.inception import InceptionResnet, Inceptionv3, Inceptionv3Attention
from models.encoders.toy_encoders import ToyEncoder, ToyAttentionEncoder, ToyAttentionOn1D
from models.encoders.vgg import VGG_16
from models.encoders.efficientnet import EfficientNetB0, EfficientNetB7, TimeDistributedEfficientNetB0
from models.encoders.vision_transformer import VisionTransformer
from models.encoders.hybrid_encoders import InceptionViT

SUPPORTED_ENCODERS = {"inceptionv1_encoder": InceptionResnet,
                      "inceptionv3_encoder": Inceptionv3,
                      "iv3_att": Inceptionv3Attention,
                      "toy_encoder": ToyEncoder,
                      "toy_att": ToyAttentionEncoder,
                      "toy_att1D": ToyAttentionOn1D,
                      "vgg16": VGG_16,
                      "effb0": EfficientNetB0,
                      "td_effb0": TimeDistributedEfficientNetB0,
                      "effb7": EfficientNetB7,
                      "ViT": VisionTransformer,
                      "IViT": InceptionViT}
