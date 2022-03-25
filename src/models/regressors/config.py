from models.regressors.dense_regressors import SUPPORTED_DENSE_REGRESSORS
from models.regressors.recurrent_regressors import SUPPORTED_RECURRENT_REGRESSORS
from models.regressors.toy_regressors import SUPPORTED_TOY_REGRESSORS

from models.regressors.permutation_regressors.permutation_regressors import SUPPORTED_PERMUTATION_REGRESSORS
from models.regressors.permutation_regressors.damonet import SUPPORTED_DAMONETS
from models.regressors.permutation_regressors.xmonet import SUPPORTED_XMONETS
from models.regressors.permutation_regressors.monet import SUPPORTED_MONET
from models.regressors.attention_regressors import SUPPORTED_ATTENTION_REGRESSORS
from models.regressors.multi_order_attention_regressors import SUPPORTED_MULTI_ORDER_ATTENTION_REGRESSORS
from models.regressors.block_attention_regressors import SUPPORTED_BLOCK_ATTENTION_REGRESSORS
from models.regressors.soft_layer_ordering import SoftLayerOrderingRegressor

from models.regressors.tta_regressors import TTAT
from models.regressors.lla_regressors import LLAT
from models.regressors.motla_regressors import MOTLAT, TLAT
from models.regressors.motcat_regressors import MOTCAT

SUPPORTED_REGRESSORS = {**SUPPORTED_DENSE_REGRESSORS,
                        **SUPPORTED_RECURRENT_REGRESSORS,
                        **SUPPORTED_TOY_REGRESSORS,
                        **SUPPORTED_PERMUTATION_REGRESSORS,
                        **SUPPORTED_DAMONETS,
                        **SUPPORTED_MONET,
                        **SUPPORTED_XMONETS,
                        **SUPPORTED_ATTENTION_REGRESSORS,
                        **SUPPORTED_MULTI_ORDER_ATTENTION_REGRESSORS,
                        **SUPPORTED_BLOCK_ATTENTION_REGRESSORS,
                        "slo": SoftLayerOrderingRegressor,
                        "ttat": TTAT,
                        "llat": LLAT,
                        "tlat": TLAT,
                        "motcat": MOTCAT,
                        "motlat": MOTLAT}
