import wandb
from main import main
from params.default_params import optimizer_param_map, default_model_params

def run_sweeps():
    run = wandb.init()
    config = wandb.config
    run.name = f"n_enc_{config.optimizer}_n_dec_{config.lr}"

    n_iter = config.n_iter
    loss = config.loss
    config.betas = (config.beta1, config.beta2)
    optimizer = optimizer_param_map[config.optimizer]
    for key in optimizer["default_params"].keys():
        optimizer["default_params"][key] = getattr(config, str(key))

    use_wandb = default_model_params["use_wandb"]
    input_embedding_size = config.input_embedding_size
    # n_encoder_layer = config.n_encoder_layer
    # n_decoder_layer = config.n_decoder_layer
    num_layer = default_model_params["num_layer"]
    hidden_size = config.hidden_size
    cell_type = config.cell_type
    bidirectional = default_model_params["bidirectional"]
    dropout = config.dropout
    teacher_forcing_ratio = config.teacher_forcing_ratio
    use_attention = default_model_params["use_attention"]

    main(n_iter, loss, optimizer, use_wandb,
         input_embedding_size, num_layer, hidden_size,
         cell_type, bidirectional, dropout, teacher_forcing_ratio, use_attention)

sweep_configuration = {
    "name": "first_rnn",
    "method": "bayes",
    "metric": {'goal': 'maximize', 'name': 'val_acc'},
    "early_terminate": {
        "type": "hyperband",
        "eta": 2,
        "min_iter": 3
     },
    "parameters": {
        "n_iter": {'min': 50000, 'max': 100000},
        'loss': {'values': ['nlll']},
        'optimizer': {'values' :['sgd', 'adam']},
        'lr': {'min': 0.001, 'max': 0.1},
        'beta1': {'min': 0.6, 'max': 0.61},
        'beta2': {'min': 0.6, 'max': 0.61},
        'momentum': {'min': 0.5, 'max': 0.9},
        "input_embedding_size": {'min': 128, 'max': 1024},
        # "n_encoder_layer": {'min': 1, 'max': 5},
        # "n_decoder_layer": {'min': 1, 'max': 5},
        "num_layer": {'min': 1, 'max': 5},
        "hidden_size": {'min': 128, 'max': 1024},
        "cell_type": {'values': ['gru', 'lstm']},
        "dropout": {'min': 0.0, 'max': 0.4},
        "teacher_forcing_ratio": {'min': 0.0, 'max': 0.5}
    }
}


sweep_id = wandb.sweep(sweep=sweep_configuration, project="cs6910-assignment-3", entity="me19b110")
wandb.agent(sweep_id=sweep_id, function=run_sweeps, count=1)
