# Assignment 3 Repository for CS6910

This repository contains all the code related to assignment 2 of cs6910.<br>

## API Reference

### Creating Model
The command below will create the model with the given params.
```python
         model = Transliterator(loss, optimizer, use_wandb,
                              input_embedding_size, num_layer,
                              hidden_size, cell_type, bidirectional, dropout,
                              teacher_forcing_ratio, use_attention, DEVICE, batch_size, train_iterator,
                              source_field, target_field, valid_iterator, valid_dataset, epochs).to(DEVICE)
```


### Training Model
The command below will train the model object for the datasets already passed
```python
model.fit()
```

### Getting Predictions
The command below will output predicted word for a single instance of test sample passed to the dataset
```python
model.predict(x_test)
```

### Getting word level accuracy
The command below will run validation tests over valid_dataset passed during creation
```python
 val_acc = eng2hin.validate()
```

## Arguments
### Loss functions
```python
    "cross_entropy": torch.nn.CrossEntropyLoss()
```

### Cell Types
```python
   "LSTM": nn.LSTM()
   "RNN": nn.RNN()
   "GRU": nn.GRU()
```

### Optimizers
```python
    SGD: torch.optim.SGD()
    Adam: torch.optim.Adam()
```

## User Interface/Scripts that can be run
| Commands | Functions |
| --- | --- |
|```python src/train.py``` | fetches parameters passed by command line and trains the model by calling ```main``` function from ```main.py``` file. This further passes arguments to the model to train it. |
| | |
| ```python src/wandb_sweep``` | contains sweep configuration details.  NOTE: Bad parameters, like Batch Size = <some small value>, have not been kept in sweep configurations in order to reduce runs and hence hyperparameter tuning time |
| | |

NOTE: type command ```python src/train.py --help``` for more information about the arguments that can be passed. You should expect output like the one below.
```python
    usage: train [-h] [-wp {cs6910-assignment-3}] [-we {me19b110}] [-ep EPOCHS] [-l {cross_entropy}] [-o {sgd,adam}] [-lr LR] [-m MOMENTUM] [-bt1 BETA1] [-bt2 BETA2]
             [-wb {0,1}] [-ies INPUT_EMBEDDING_SIZE] [-nl NUM_LAYER] [-hs HIDDEN_SIZE] [-ct {lstm,gru}] [-bd {0,1}] [-do DROPOUT] [-tfe TEACHER_FORCING_RATIO]
             [-ua {0,1}] [-bs BATCH_SIZE] [-sm {0,1}]

Supply parameters to Encoder Decoder architecture to run and log results in wandb.ai

options:
  -h, --help            show this help message and exit
  -wp {cs6910-assignment-3}, --wandb_project {cs6910-assignment-3}
  -we {me19b110}, --wandb_entity {me19b110}
  -ep EPOCHS, --epochs EPOCHS
  -l {cross_entropy}, --loss {cross_entropy}
  -o {sgd,adam}, --optimizer {sgd,adam}
  -lr LR, --lr LR
  -m MOMENTUM, --momentum MOMENTUM
  -bt1 BETA1, --beta1 BETA1
  -bt2 BETA2, --beta2 BETA2
  -wb {0,1}, --use_wandb {0,1}
  -ies INPUT_EMBEDDING_SIZE, --input_embedding_size INPUT_EMBEDDING_SIZE
  -nl NUM_LAYER, --num_layer NUM_LAYER
  -hs HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
  -ct {lstm,gru}, --cell_type {lstm,gru}
  -bd {0,1}, --bidirectional {0,1}
  -do DROPOUT, --dropout DROPOUT
  -tfe TEACHER_FORCING_RATIO, --teacher_forcing_ratio TEACHER_FORCING_RATIO
  -ua {0,1}, --use_attention {0,1}
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
  -sm {0,1}, --save_model {0,1}
```
    
## Default arguments and Description

| Name | Default Value | Description |
| --- | --- | --- |
| -wp, --wandb_project |	cs6910-assignment-1 |	Project name used to track experiments in Weights & Biases dashboard |
| -we, --wandb_entity	| me19b110 |	Wandb Entity used to track experiments in the Weights & Biases dashboard |
| -ep, --epochs |	32 | Number of epochs |
| -l, --loss |	"cross_entropy" | which loss criteria to select for training model |
| -o, --optimizer |	"adam" |	choices: ["sgd", "adam"] |
| -lr, --learning_rate |	0.001 |	Learning rate used to optimize model parameters |
| -m, --momentum | 0.01	| momentum for sgd optimizer |
| -bt1, --beta1 | 0.8	| beta1 for adam optimizer |
| -bt2, --beta2 | 0.8	| beta2 for adam optimizer |
| -wb, --use_wandb | 1 | choices: [0, 1]  |
| -ips, --input_embedding_size | 256 | embedding size for input and output languages  |
| -nl, --num_layer | 3 | number of layers for stacking rnns |
| -hs, --hidden_size | 512 | hidden size for rnns |
| -ct, --cell_type | "lstm" | choices=["lstm", "rnn", "gru"] |
| -bd, --bidirectional | 0 | weather to train model in bidirectional fashion or not |
| -do, --dropout | 0.3 | dropout fraction for embedding and cells |
| -tfe, --teacher_forcing_ratio | 0.3 | whether to perform teacher forcing or not |
| -ua, --use_attention | 0 | whether to use attention or not |
| -bs, --batch_size | 1024 | batch size to be used |
| -sm, --save_model | 0 | whether to save model or not |

## Contributors

student name: HARSHIT RAJ  
email: me19b110@smail.iitm.ac.in  
 
course: CS6910 - FUNDAMENTALS OF DEEP LEARNING  
professor: DR. MITESH M. KHAPRA  
 
ta: ASHWANTH KUMAR  
email: cs21m010@smail.iitm.ac.in   
