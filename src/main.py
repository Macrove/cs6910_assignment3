import os
import torch
import time
import matplotlib.pyplot as plt
import wandb
from utils.env import TRAIN_FILENAME, TEST_FILENAME, VALID_FILENAME, DIR_PATH, LANG1NAME, LANG2NAME
from seq2seq.nn import Transliterator
from utils.env import DEVICE
from torchtext.data import Field, TabularDataset, BucketIterator
import pandas as pd

train_file = os.path.join(DIR_PATH, TRAIN_FILENAME)
valid_file = os.path.join(DIR_PATH, VALID_FILENAME)
test_file = os.path.join(DIR_PATH, TEST_FILENAME)

source_field = Field(tokenize=lambda x: list(x), lower=True, include_lengths=True, init_token='<sos>', eos_token='<eos>', pad_token='<pad>')
target_field = Field(tokenize=lambda x: list(x), lower=True, include_lengths=True, init_token='<sos>', eos_token='<eos>', pad_token='<pad>')

# Define the fields and their order in the CSV/TSV file
fields = [('src', source_field), ('trg', target_field)]

train_dataset = TabularDataset(train_file, 'csv', fields=fields, skip_header=False)
valid_dataset = TabularDataset(valid_file, 'csv', fields=fields, skip_header=False)
test_dataset = TabularDataset(test_file, 'csv', fields=fields, skip_header=False)

source_field.build_vocab(train_dataset)
target_field.build_vocab(train_dataset)

def main(loss, optimizer, use_wandb, input_embedding_size,
         num_layer, hidden_size, cell_type, bidirectional, 
         dropout, teacher_forcing_ratio, use_attention, batch_size, epochs, save_model):

     #making iterators
     train_iterator = BucketIterator(train_dataset, batch_size=batch_size, device=DEVICE, sort_key=lambda x: len(x.src), sort_within_batch=True)
     valid_iterator = BucketIterator(valid_dataset, batch_size=batch_size, device=DEVICE, sort_key=lambda x: len(x.src), sort_within_batch=True)

     eng2hin = Transliterator(loss, optimizer, use_wandb,
                              input_embedding_size, num_layer,
                              hidden_size, cell_type, bidirectional, dropout,
                              teacher_forcing_ratio, use_attention, DEVICE, batch_size, train_iterator,
                              source_field, target_field, valid_iterator, valid_dataset, epochs).to(DEVICE)

     #training model
     start = time.time()
     eng2hin.fit()
     end = time.time()

     print("Total training time: ", (end - start), "s")
    
     # accuracy on validation data
     val_acc, _ = eng2hin.validate(valid_dataset)
     val_acc *= 100
     print(f"Word level accuracy : {val_acc:.4f} %\n")
     
     if save_model == True:
          test_acc, pred_target_pairs = eng2hin.validate(test_dataset, return_preds=True)
          test_acc*=100
          print("Best Test Accuracy: ", test_acc)
          df = pd.DataFrame(pred_target_pairs, columns=["pred_word", "target_word", "isCorrect"])
          df.to_csv("./predictions_vanilla/predictions.csv", index=False)

          correct_predictions = [x for x in pred_target_pairs if x[2] == True][:20]
          incorrect_predictions = [x for x in pred_target_pairs if x[2] == False][:20]

          correct_df = pd.DataFrame(correct_predictions, columns=["Predicted Word", "Target Word", "Correct"])
          incorrect_df = pd.DataFrame(incorrect_predictions, columns=["Predicted Word", "Target Word", "Correct"])

          correct_table = wandb.Table(dataframe=correct_df)
          incorrect_table = wandb.Table(dataframe=incorrect_df)

          wandb.log({"Correct Predictions": correct_table,
           "Incorrect Predictions": incorrect_table})



          torch.save(eng2hin.state_dict(), f"./model_{val_acc:.2f}_{num_layer}_{dropout}_{input_embedding_size}_{epochs}")
          

     if use_wandb:
          wandb.log({"val_acc": val_acc})

     