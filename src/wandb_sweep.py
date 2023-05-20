import wandb
from main import main
from params.default_params import optimizer_param_map, default_model_params

def run_sweeps():
    run = wandb.init()
    config = wandb.config

    loss = config.loss
    config.betas = (config.beta1, config.beta2)
    optimizer = optimizer_param_map[config.optimizer]
    for key in optimizer["default_params"].keys():
        optimizer["default_params"][key] = getattr(config, str(key))

    use_wandb = 1
    input_embedding_size = config.input_embedding_size
    num_layer = config.num_layer
    hidden_size = config.hidden_size
    cell_type = config.cell_type
    bidirectional = default_model_params["bidirectional"]
    dropout = config.dropout
    teacher_forcing_ratio = config.teacher_forcing_ratio
    use_attention = default_model_params["use_attention"]
    batch_size = default_model_params["batch_size"]
    epochs = config["epochs"]
    save_model = default_model_params["save_model"]
    run.name = f"nl_{num_layer}_hs_{hidden_size}"

    main(loss, optimizer, use_wandb,
         input_embedding_size, num_layer, hidden_size,
         cell_type, bidirectional, dropout, teacher_forcing_ratio, use_attention, batch_size, epochs, save_model)

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
        'loss': {'values': ['cross_entropy']},
        'optimizer': {'values' :['sgd', 'adam']},
        'lr': {'min': 0.001, 'max': 0.9},
        'beta1': {'min': 0.5, 'max': 0.9},
        'beta2': {'min': 0.5, 'max': 0.9},
        'momentum': {'min': 0.0, 'max': 0.9},
        "input_embedding_size": {'min': 128, 'max': 512},
        "num_layer": {'min': 1, 'max': 5},
        "hidden_size": {'min': 128, 'max': 512},
        "cell_type": {'values': ['gru', 'lstm', 'rnn']},
        "dropout": {'min': 0.0, 'max': 0.4},
        "teacher_forcing_ratio": {'min': 0.0, 'max': 0.5},
        "epochs": {'min': 8, 'max': 12}
    }
}


sweep_id = wandb.sweep(sweep=sweep_configuration, project="cs6910-assignment-3", entity="me19b110")
wandb.agent(sweep_id=sweep_id, function=run_sweeps, count=1)
