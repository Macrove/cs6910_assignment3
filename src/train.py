import argparse
import wandb
from params.default_params import default_credentials, default_model_params, optimizer_param_map
from params.parser_params import parser_choices
from main import main


parser = argparse.ArgumentParser(
                    prog='train',
                    description='Supply parameters to Encoder Decoder architecture\
                        to run and log results in wandb.ai',
                    epilog="That's all")

optimizer = optimizer_param_map[default_model_params["optimizer"]]

parser.add_argument("-wp", "--wandb_project", choices= parser_choices["wandb_project"], default=default_credentials["wandb_project"])
parser.add_argument("-we", "--wandb_entity", choices=parser_choices["wandb_entity"], default=default_credentials["wandb_entity"])
parser.add_argument("-ep", "--epochs", default=default_model_params["epochs"], type=int)
parser.add_argument("-l", "--loss", choices= parser_choices["loss"], default=default_model_params["loss"])
parser.add_argument("-o", "--optimizer", choices=parser_choices["optimizer"], default=default_model_params["optimizer"])
parser.add_argument("-lr", "--lr", default=optimizer["default_params"]["lr"], type=float)
parser.add_argument("-m", "--momentum", default=optimizer_param_map["sgd"]["default_params"]["momentum"], type=float)
parser.add_argument("-bt1", "--beta1", default=optimizer_param_map["adam"]["default_params"]["betas"][0], type=float)
parser.add_argument("-bt2", "--beta2", default=optimizer_param_map["adam"]["default_params"]["betas"][1], type=float)
parser.add_argument("-wd", "--weight_decay", default=optimizer_param_map["sgd"]["default_params"]["weight_decay"], type=float)
parser.add_argument("-wb", "--use_wandb", choices=parser_choices["use_wandb"], default=default_model_params["use_wandb"], type=int)
parser.add_argument("-ies", "--input_embedding_size", default=default_model_params["input_embedding_size"], type=int)
parser.add_argument("-nl", "--num_layer", default=default_model_params["num_layer"], type=int)
parser.add_argument("-hs", "--hidden_size", default=default_model_params["hidden_size"], type=int)
parser.add_argument("-ct", "--cell_type", choices=parser_choices["cell_type"], default=default_model_params["cell_type"])
parser.add_argument("-bd", "--bidirectional", choices=parser_choices["bidirectional"], default=default_model_params["bidirectional"])
parser.add_argument("-do", "--dropout", default=default_model_params["dropout"])
parser.add_argument("-tfe", "--teacher_forcing_ratio", default=default_model_params["teacher_forcing_ratio"])
parser.add_argument("-ua", "--use_attention", choices=parser_choices["use_attention"], default=default_model_params["use_attention"], type=int)
parser.add_argument("-bs", "--batch_size", default=default_model_params["batch_size"], type=int)
args = parser.parse_args()
print(args)

epochs = args.epochs
loss = args.loss
args.betas = (args.beta1, args.beta2)

optimizer = optimizer_param_map[args.optimizer]
for key in optimizer["default_params"].keys():
    optimizer["default_params"][key] = getattr(args, str(key))

print(optimizer)

lr = args.lr
betas = args.betas
momentum = args.momentum
use_wandb = args.use_wandb
input_embedding_size = args.input_embedding_size
num_layer = args.num_layer
hidden_size = args.hidden_size
cell_type = args.cell_type
bidirectional = args.bidirectional
dropout = args.dropout
teacher_forcing_ratio = args.teacher_forcing_ratio
use_attention = args.use_attention
batch_size = args.batch_size

if use_wandb:
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    run.name = "ac_{}_opt_{}".format(args.activation, args.optimizer)
    wandb.log({
        "epochs": args.epochs,
        "loss": args.loss,
        "optimizer": args.optimizer.name,
        "lr": args.lr,
        "beta1": betas[0],
        "beta2": betas[1],
        "momentum": args.momentum,
        "use_wandb": args.use_wandb,
        "input_embedding_size": args.input_embedding_size,
        "num_layer": args.num_layer,
        "hidden_size": args.hidden_layer_size,
        "cell_type": args.cell_type,
        "bidirectional": args.bidirectional,
        "dropout": args.dropout,
        "teacher_forcing_ratio": args.teacher_forcing_ratio,
        "use_attention": args.use_attention,
        "batch_size": args.batch_size
    })
    run.log_code()

if __name__ == '__main__':
    main(loss, optimizer, use_wandb,
         input_embedding_size, num_layer, hidden_size,
         cell_type, bidirectional, dropout, teacher_forcing_ratio, use_attention, batch_size, epochs)