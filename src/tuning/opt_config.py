CONSTANT_OPT_REQUIRED_PARAMS = {"adam": {"epsilon": "epsilon"},
                                "adamw": {"epsilon": "epsilon"}}

SCHEDULABLE_OPT_REQUIRED_PARAMS = {"adamw": {"learning_rate": "lr",
                                             "weight_decay": "wd"},
                                   "adam": {"learning_rate": "lr"},
                                   "sgd": {"learning_rate": "lr"}}

SCHEDULE_REQUIRED_PARAMS = {"exponential": ["decay"],
                            "pw_constant": ["val1", "cut", "val2"],
                            "mt_transformer": ["lin_phase"],
                            "vit_transformer": ["lin_phase"],
                            "warmup_exponential": ["init", "warmup", "decay"],
                            "mt_pretraining": ["warmup", "warmup_value", "lin_phase"],
                            "mt_pretraining_v2": ["warmup", "lin_phase1", "lin_phase2"],
                            "vit_pretraining": ["warmup", "lin_phase"],
                            "controller_pretraining": ["warmup", "lin_phase"]}

EXTERNAL_SCHEDULE_REQUIRED_PARAMS = {"mt_transformer": ["d_model_y", "batchsize", "T"],
                                     "vit_transformer": ["d_model_x", "batchsize", "num_patches"],
                                     "mt_pretraining": ["d_model_y", "batchsize", "T"],
                                     "mt_pretraining_v2": ["d_model_y", "batchsize", "T"],
                                     "vit_pretraining": ["d_model_x", "batchsize", "num_patches"],
                                     "controller_pretraining": ["d_model_y", "batchsize", "T"]
                                     }

CONDITION_REQUIRED_PARAMS = {"above_th": ["threshold"],
                             "below_th": ["threshold"]}
