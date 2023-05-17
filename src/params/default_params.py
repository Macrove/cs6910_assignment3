default_model_params = {
    "n_iter": 75000,
    "loss": "nlll",
    "optimizer": "sgd",
    "lr": 0.01,
    "use_wandb": 0,
    "is_bidirectional": False,
    "input_embedding_size": 256,
    # "n_encoder_layer": 2,
    # "n_decoder_layer": 3,
    "num_layer": 2,
    "hidden_size": 128,
    "cell_type": "gru",
    "bidirectional": 0,
    "dropout": 0.2,
    "teacher_forcing_ratio": 0.5,
    "use_attention": 1
}

default_credentials = {
    "wandb_project": "cs6910-assignment-3",
    "wandb_entity": "me19b110"
}

optimizer_param_map = {
    "sgd" : {
        "name": "sgd",
        "default_params": dict(
            lr = 0.01,
            momentum= 0.01,
        )
    },
    "adam" : {
        "name": "adam",
        "default_params": dict(
            lr= 0.01,
            betas= (0.9, 0.99),
        )
    }
}