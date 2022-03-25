CONSTANT_OPT_REQUIRED_PARAMS = {"adam": {"epsilon": "epsilon"},
                                "adamw": {"epsilon": "epsilon"}}

SCHEDULABLE_OPT_REQUIRED_PARAMS = {"adamw": {"learning_rate": "lr",
                                             "weight_decay": "wd"},
                                   "adam": {"learning_rate": "lr"},
                                   "sgd": {"learning_rate": "lr"}}

SCHEDULE_REQUIRED_PARAMS = {"exponential": ["decay"],
                            "mt_transformer": ["lin_phase"],
                            "vit_transformer": ["lin_phase"]}

EXTERNAL_SCHEDULE_REQUIRED_PARAMS = {"mt_transformer": ["d_model_y", "batchsize", "T"],
                                     "vit_transformer": ["d_model_x", "batchsize", "num_patches"]}

CONDITION_REQUIRED_PARAMS = {"above_th": ["threshold"],
                             "below_th": ["threshold"]}
