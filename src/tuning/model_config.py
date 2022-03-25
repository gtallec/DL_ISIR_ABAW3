TEST_PARAMS = {"recurrent_units": ['recurrent_cell', 'units'],
               "recurrent_activation": ['recurrent_cell', 'activation']}

# ENCODERS: 
TOY_ENCODER_PARAMS = {"encoder_bottleneck": "bottleneck"}
INCEPTION_ENCODER_PARAMS = {"encoder_bottleneck": "bottleneck_layer_size"}
INCEPTIONV3_ENCODER_PARAMS = {"pooling": "pooling"}
INCEPTIONV3_ATTENTION_PARAMS = {"d_model_x": "d_model"}

TIME_DISTRIBUTED_EFFICIENTNET = {"S": "S",
                                 "pooling": "pooling",
                                 "bottleneck": "bottleneck"}
# REGRESSORS:
VMNS_PARAMS = {"units": "units",
               "imbalance": "imbalance"}

TVMNS = {"units": "units",
         "S": "S"}

VMNC_PARAMS = {"units": "units"}
MRNN_PARAMS = {"units": ["recurrent_cell_args", "units"],
               "N_sample": "N_sample"}

MONET_PARAMS = {"drop_out": "drop_out",
                "N_sample": "N_sample",
                "n_permutations": "n_permutations",
                "recurrent_units": ["recurrent_cell_args", "units"]}
PIM2SEQ_PARAMS = {"N_sample": "N_sample",
                  "n_permutations": "n_permutations",
                  "recurrent_units": ["recurrent_cell_args", "units"]}

GMONET_PARAMS = {"N_sample": "L",
                 "n_permutations": "P",
                 "dropout": "k",
                 "recurrent_units": ["recurrent_cell_args", "units"],
                 "label_units": "Lu"}

SINGLE_MONET_PARAMS = {"recurrent_units": ["recurrent_cell_args", "units"],
                       "order": "order",
                       "N_sample": "N_sample"}

HARD_MONET_PARAMS = {"recurrent_units": ["recurrent_cell_args", "units"],
                     "N_sample": "N_sample",
                     "n_permutations": "n_permutations",
                     "label_units": "label_units"}

ATTENTION_REGRESSOR = {"num_layers": "num_layers",
                       "d_model": "d_model",
                       "num_heads": "num_heads",
                       "dff": "dff",
                       "rate": "rate",
                       "label_temperature": "label_temperature",
                       "use_encoder": "use_encoder",
                       "mask_start_token": "mask_start_token"}


UATTENTION_REGRESSOR = {"num_layers": "num_layers",
                        "d_model": "d_model",
                        "num_heads": "num_heads",
                        "dff": "dff",
                        "rate": "rate"}


RUATTENTION_REGRESSOR = {"num_layers": "num_layers",
                         "d_model": "d_model",
                         "num_heads": "num_heads",
                         "dff": "dff",
                         "rate": "rate"}

MULTITASKTRANSFORMER = {"num_layers": "num_layers",
                        "d_model": "d_model",
                        "num_heads": "num_heads",
                        "dff": "dff",
                        "rate": "rate",
                        "temp_tx": "temp_tx",
                        "temp_ty": "temp_ty",
                        "use_encoder": "use_encoder",
                        "use_labels": "use_labels"}

MOMTTRANSFORMER = {"num_layers": "num_layers",
                   "d_model": "d_model",
                   "num_heads": "num_heads",
                   "M": "M",
                   "dff": "dff",
                   "rate": "rate",
                   "temp_tx": "temp_tx",
                   "temp_ty": "temp_ty",
                   "use_encoder": "use_encoder",
                   "use_labels": "use_labels",
                   "beam_search": "beam_search",
                   "beam_width": "beam_width",
                   "shared_dense": "shared_dense",
                   "permutation_encoding": "permutation_encoding"}


LLAT = {"num_layers": "num_layers",
        "d_model": "d_model",
        "num_heads": "num_heads",
        "dff": "dff",
        "rate": "rate",
        "temp_yy": "temp_yy",
        "temp_yx": "temp_yx",
        "attention_mode": "attention_mode",
        "pred_mode": "pred_mode",
        "pred_N": "pred_N",
        "shared_dense": "shared_dense"}

TTAT = {"num_layers": "num_layers",
        "d_model": "d_model",
        "num_heads": "num_heads",
        "mlp_scale": "mlp_scale",
        "rate": "rate",
        "temp_tx": "temp_tx",
        "temp_tt": "temp_tt"}

TLAT = {"num_layers": "num_layers",
        "d_model": "d_model",
        "num_heads": "num_heads",
        "dff": "dff",
        "rate": "rate",
        "temp_ty": "temp_ty",
        "temp_tx": "temp_tx",
        "pred_mode": "pred_mode",
        "pred_N": "pred_N",
        "shared_dense": "shared_dense"}

MOTLAT = {"num_layers": "num_layers",
          "d_model": "d_model",
          "num_heads": "num_heads",
          "M": "M",
          "dff": "dff",
          "rate": "rate",
          "temp_tx": "temp_tx",
          "temp_ty": "temp_ty",
          "pred_mode": "pred_mode",
          "pred_N": "pred_N",
          "shared_dense": "shared_dense",
          "permutation_heuristic": "permutation_heuristic",
          "permutation_encoding": "permutation_encoding"}

MOTCAT = {"num_layers": "num_layers_t",
          "num_layers_y": "num_layers_y",
          "d_model": "d_model_t",
          "d_model_y": "d_model_y",
          "num_heads": "num_heads_t",
          "num_heads_y": "num_heads_y",
          "mlp_scale": "mlp_scale_t",
          "mlp_scale_y": "mlp_scale_y",
          "rate": "rate_t",
          "rate_y": "rate_y",
          "ca_order": "ca_order",
          "M": "M",
          "permutation_heuristic": "permutation_heuristic",
          "rate_corrupt": "rate_corrupt",
          "temp_tx": "temp_tx",
          "temp_ty": "temp_ty",
          "temp_yy": "temp_yy",
          "pred_N": "pred_N"}

VIT = {"d_model_x": "d_model",
       "num_heads_x": "num_heads",
       "mlp_scale_x": "mlp_scale",
       "num_layers_x": "num_layers",
       "rate_x": "rate",
       "temp_xx": "temp_xx",
       "num_patches": "num_patches"}

IVIT = {"d_model_x": "d_model",
        "num_heads_x": "num_heads",
        "mlp_scale_x": "mlp_scale",
        "num_layers_x": "num_layers",
        "rate_x": "rate",
        "temp_xx": "temp_xx"}


BMOMTTRANSFORMER = {"num_layers": "num_layers",
                    "d_model": "d_model",
                    "num_heads": "num_heads",
                    "M": "M",
                    "dff": "dff",
                    "rate": "rate",
                    "blocksize": "blocksize",
                    "temp_tx": "temp_tx",
                    "temp_ty": "temp_ty",
                    "pred_mode": "pred_mode",
                    "pred_N": "pred_N",
                    "shared_dense": "shared_dense",
                    "permutation_encoding": "permutation_encoding"}



# IMAGE TRANSFORMER
IMAGE_TRANSFORMER_PARAMS = {"num_layers": "num_layers",
                            "d_model": "d_model",
                            "num_heads": "num_heads",
                            "dff": "dff",
                            "num_patches": "num_patches",
                            "patch_size": "patch_size",
                            "rate": "rate"}

MONETv2_PARAMS = {"drop_out": "drop_out",
                  "N_sample": "N_sample",
                  "n_permutations": "n_permutations",
                  "recurrent_units": ["recurrent_cell_args", "units"],
                  "label_units": "label_units",
                  "permutation_encoding": "permutation_encoding",
                  "permutation_units": "permutation_units"}

XMONET_PARAMS = {"dropout": "dropout",
                 "N_sample": "N_sample",
                 "recurrent_units": ["recurrent_cell_args", "units"],
                 "permutation_encoding": "permutation_encoding"}

XMONET_CATEGORICAL_PARAMS = {"dropout": "k",
                             "n_perm": "P",
                             "N_sample": "N_sample",
                             "recurrent_units": ["recurrent_cell_args", "units"],
                             "permutation_encoding": "permutation_encoding"}


MAONET_PARAMS = {"dropouts": "dropouts",
                 "N_samples": "N_samples",
                 "mixtX": "mixtX",
                 "maonet_units": ["recurrent_cell_args", "units"]}

DAMONET_PARAMS = {"dropouts": "dropouts",
                  "N_samples": "N_samples",
                  "damonet_units": ["recurrent_cell_args", "units"],
                  "pred_units": "pred_units",
                  "routing_units": "routing_units"}
DAMONETV1_PARAMS = {"dropouts": "dropouts",
                    "N_samples": "N_samples",
                    "damonet_units": ["recurrent_cell_args", "units"],
                    "pred_units": "pred_units",
                    "label_units": "label_units",
                    "mixtX": "mixtX"}
                    

# MIXTURES:
VECTOR_PARAMS = {"n_permutations": "n_permutations"}
RECURRENT_MIXTURE_PARAMS = {"mixture_units": ["recurrent_cell_args", "units"]}
XMIXTURE_PARAMS = {"n_perm": "P",
                   "mixture_units": "units",
                   "monet_init": "uniform_init"}

SOFT_LAYER_ORDERING_PARAMS = {"depth": "D",
                              "n_modules": "M",
                              "modules_units": "M_units",
                              "modules_bias": "M_use_bias"}

# CONTROLLERS:
CONTROLLERS_PARAMS = {"d_model": "d_model",
                      "num_layers": "num_layers"}

MODEL_REQUIRED_PARAMS = {"test": TEST_PARAMS,
                         "pim2seq": PIM2SEQ_PARAMS,
                         "toy_encoder": TOY_ENCODER_PARAMS,
                         "inceptionv1_encoder": INCEPTION_ENCODER_PARAMS,
                         "vmns": VMNS_PARAMS,
                         "tvmns": TVMNS,
                         "vmnc": VMNC_PARAMS,
                         "mrnn": MRNN_PARAMS,
                         "td_effb0": TIME_DISTRIBUTED_EFFICIENTNET,
                         "inceptionv3_encoder": INCEPTIONV3_ENCODER_PARAMS,
                         "iv3_att": INCEPTIONV3_ATTENTION_PARAMS,
                         "dospim2seq": MONET_PARAMS,
                         "gmonet": GMONET_PARAMS,
                         "monet": MONETv2_PARAMS,
                         "imonet": MONETv2_PARAMS,
                         "ruattention_regressor": RUATTENTION_REGRESSOR,
                         "uattention_regressor": UATTENTION_REGRESSOR,
                         "mtt": MULTITASKTRANSFORMER,
                         "momtt": MOMTTRANSFORMER,
                         "bmomtt": BMOMTTRANSFORMER,
                         "attention_regressor": ATTENTION_REGRESSOR,
                         "image_transformer": IMAGE_TRANSFORMER_PARAMS,
                         "single_monet": SINGLE_MONET_PARAMS,
                         "xmonet": XMONET_PARAMS,
                         "xmonet_ext": XMONET_PARAMS,
                         "vector": VECTOR_PARAMS,
                         "maonet": MAONET_PARAMS,
                         "damonet": DAMONET_PARAMS,
                         "damonetv1": DAMONETV1_PARAMS,
                         "rec_mixture": RECURRENT_MIXTURE_PARAMS,
                         "rec_mixturev2": RECURRENT_MIXTURE_PARAMS,
                         "slo": SOFT_LAYER_ORDERING_PARAMS,
                         "hard_monet": HARD_MONET_PARAMS,
                         "xmixture": XMIXTURE_PARAMS,
                         "xmonet_categorical": XMONET_CATEGORICAL_PARAMS,
                         "controllers": CONTROLLERS_PARAMS,
                         "tlat": TLAT,
                         "motlat": MOTLAT,
                         "motcat": MOTCAT,
                         "ttat": TTAT,
                         "llat": LLAT,
                         "ViT": VIT,
                         "IViT": IVIT}
