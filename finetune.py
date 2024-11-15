# Finetune AmyPETNet model.

from DeepPET.model import DeepPETModelManager
from DeepPET.architecture import DeepPETEncoderGradCAM
from DeepPET.data import DeepPETDataGenerator
import argparse
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

parser = argparse.ArgumentParser(description='DeepPET model testing')
parser.add_argument('data_dir',
                        help='path to save/find extracted and downloaded scans')  
parser.add_argument('--metadata', nargs='?', required=True,
                        help='path to metadata')  # assumes csv file is within data_dir if unspecified 
parser.add_argument('--target_col', type=str,  required=True, help='Name of target column in metadata.')
parser.add_argument('--ids', nargs='*',
                        help='text or list of IDs to include for training.')

args = parser.parse_args()
target_col = args.target_col

def read_from_txt(filepath):
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    return lines

subject_ids = read_from_txt(args.ids[0]) if os.path.exists(args.ids[0]) else args.ids 

# Load in model
model = DeepPETEncoderGradCAM()
model_manager = DeepPETModelManager(model=model, odir='model', checkpoint="model/model.pth")

# Fine tuning

# Create dataloader from AMYPAD ds
ft_df = pd.read_csv(args.metadata)
fpaths = [os.path.join(args.data_dir, f) for f in ft_df['img_path'].values.flatten()] 

ft_datagen = DeepPETDataGenerator(
    fpaths=fpaths,
    labels=ft_df[target_col].values,
)

# split into training and validation indexes
train_idxs, val_idxs = train_test_split(range(len(subject_ids)), test_size=0.5, random_state=0) 
train_ids = [subject_ids[i] for i in train_idxs]

ft_ds = ft_datagen.create_dataset(cache_dir="/tmp", idx=train_idxs, mode="training")
ft_val_ds = ft_datagen.create_dataset(cache_dir="/tmp", idx=val_idxs, mode="validation")

# train the model
model_manager.train_model(ft_ds, ft_val_ds, loss_function=BCEWithLogitsLoss(), optimizer=Adam(model.parameters()), num_epochs=20, batch_size=4)
