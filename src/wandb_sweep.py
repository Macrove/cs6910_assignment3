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
    "name": "narrow_lstm",
    "method": "bayes",
    "metric": {'goal': 'maximize', 'name': 'val_acc'},
    "early_terminate": {
        "type": "hyperband",
        "eta": 2,
        "min_iter": 3
     },
    "parameters": {
        'loss': {'values': ['cross_entropy']},
        'optimizer': {'values' :['adam']},
        'lr': {'min': 0.0002, 'max': 0.0004},
        'beta1': {'min': 0.7, 'max': 0.73},
        'beta2': {'min': 0.65, 'max': 0.68},
        'momentum': {'min': 0.5, 'max': 0.6},
        "input_embedding_size": {'values': [256]},
        "num_layer": {'min': 5, 'max': 6},
        "hidden_size": {'values': [512]},
        "cell_type": {'values': ['lstm']},
        "dropout": {'min': 0.3, 'max': 0.4},
        "teacher_forcing_ratio": {'min': 0.2, 'max': 0.3},
        "epochs": {'min': 25, 'max': 30}
    }
}


sweep_id = wandb.sweep(sweep=sweep_configuration, project="cs6910-assignment-3", entity="me19b110")
wandb.agent(sweep_id=sweep_id, function=run_sweeps, count=200)
