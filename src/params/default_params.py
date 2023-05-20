default_model_params = {
    "batch_size": 64,
    "epochs": 10,
    "loss": "nlll",
    "optimizer": "sgd",
    "use_wandb": 0,
    "is_bidirectional": False,
    "input_embedding_size": 128,
    "num_layer": 2,
    "hidden_size": 128,
    "cell_type": "lstm",
    "bidirectional": 0,
    "dropout": 0.2,
    "teacher_forcing_ratio": 0.5,
    "use_attention": 0,
}

default_credentials = {
    "wandb_project": "cs6910-assignment-3",
    "wandb_entity": "me19b110"
}

optimizer_param_map = {
    "sgd" : {
        "name": "sgd",
        "default_params": dict(
           lr = 0.5,
            momentum= 0.5,
            weight_decay=0.0
        )
    },
    "adam" : {
        "name": "adam",
        "default_params": dict(
            lr= 0.001,
            betas= (0.7, 0.8),
        )
    }
}