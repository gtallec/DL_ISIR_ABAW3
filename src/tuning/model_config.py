
# ENCODERS: 
INCEPTIONV3_ENCODER_PARAMS = {"pooling": "pooling"}
INCEPTIONV3_ATTENTION_PARAMS = {"d_model_x": "d_model"}
VIT = {"d_model_x": "d_model",
       "num_heads_x": "num_heads",
       "mlp_scale_x": "mlp_scale",
       "num_layers_x": "num_layers",
       "rate_x": "rate",
       "temp_xx": "temp_xx",
       "num_patches": "num_patches"}

# REGRESSORS:
TTAT = {"num_layers": "num_layers",
        "d_model": "d_model",
        "num_heads": "num_heads",
        "mlp_scale": "mlp_scale",
        "rate": "rate",
        "temp_tx": "temp_tx",
        "temp_tt": "temp_tt"}

MODEL_REQUIRED_PARAMS = {"inceptionv3_encoder": INCEPTIONV3_ENCODER_PARAMS,
                         "iv3_att": INCEPTIONV3_ATTENTION_PARAMS,
                         "ttat": TTAT,
                         "ViT": VIT}

