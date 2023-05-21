default_model_params = {
    "batch_size": 512,
    "epochs": 1,
    "loss": "cross_entropy",
    "optimizer": "adam",
    "use_wandb": 0,
    "is_bidirectional": False,
    "input_embedding_size": 256,
    "num_layer": 1,
    "hidden_size": 512,
    "cell_type": "lstm",
    "bidirectional": 0,
    "dropout": 0,
    "teacher_forcing_ratio": 0.5,
    "use_attention": 1,
    "save_model": 0
}

default_credentials = {
    "wandb_project": "cs6910-assignment-3",
    "wandb_entity": "me19b110"
}

optimizer_param_map = {
    "sgd" : {
        "name": "sgd",
        "default_params": dict(
           lr = 0.75,
            momentum= 0.5,
        )
    },
    "adam" : {
        "name": "adam",
        "default_params": dict(
            lr= 0.000323,
            betas= (0.7120, 0.6583),
        )
    }
}