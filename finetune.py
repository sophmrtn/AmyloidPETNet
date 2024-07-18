# Finetune AmyPETNet using visual read labels and AMYPAD dataset.

from DeepPET.model import DeepPETModelManager
from DeepPET.architecture import DeepPETEncoderGradCAM
from DeepPET.data import DeepPETDataGenerator
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW

parser = argparse.ArgumentParser(description='DeepPET model testing')
parser.add_argument('data_dir',
                        help='path to save/find extracted and downloaded scans')  
parser.add_argument('--metadata', nargs='?',
                        help='path to metadata')  # assumes csv file is within data_dir if unspecified 
parser.add_argument('--ids', nargs='*',
                        help='list of IDs to include for training. Example usage: --ids 1 2 3 ')
args = parser.parse_args()

# Load in model
model = DeepPETEncoderGradCAM()
model_manager = DeepPETModelManager(model=model, odir='model')

# Fine tuning

# Create dataloader from AMYPAD ds
ft_df = pd.read_csv(args.metadata)
# filter by test ids
ft_df = ft_df[ft_df['amypad_id'].isin(args.ids)]

ft_datagen = DeepPETDataGenerator(
    args.data_dir,
    fpaths=args.ids,
    labels=ft_df['pet_vr_classification'].values,
)

# split into training and validation by ids
train_ids, val_ids = train_test_split(range(len(args.ids)), test_size=0.5, random_state=0) 

ft_ds = ft_datagen.create_dataset(cache_dir="/tmp", idx=train_ids, mode="training")
ft_val_ds = ft_datagen.create_dataset(cache_dir="/tmp", idx=val_ids, mode="validation")

# pos_weight
pos_w = ft_df.query('amypad_id in @train_ids & pet_vr_classification==0').shape[0]/ft_df.query('amypad_id in @train_ids & pet_vr_classification==1').shape[0]

# train the model
model_manager.train_model(ft_ds, ft_val_ds, loss_function=BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w])), optimizer=AdamW(model.parameters(), lr=0.01), num_epochs=20, batch_size=16)
