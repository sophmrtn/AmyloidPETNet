import argparse
import os
import math
import pandas as pd
from DeepPET.data import DeepPETDataGenerator
from DeepPET.architecture import DeepPETEncoderGradCAM
from DeepPET.model import DeepPETModelManager

# initialize parser
parser = argparse.ArgumentParser(description='DeepPET model testing')
parser.add_argument('data_dir',
                        help='path to save/find extracted and downloaded scans')  

parser.add_argument('--odir', help='model directory')
parser.add_argument('--metadata', nargs='?',
                        help='path to metadata')  # assumes csv file is within data_dir if unspecified
parser.add_argument('--ids', nargs='*',
                        help='list of IDs to include for training. Example usage: --ids 1 2 3 ')      # default value is None
args = parser.parse_args()

# parse arguments 
odir = str(args.odir)
print(f"model directory: {odir}")
ds_path = str(args.metadata)
print(f"path to testing dataset: {ds_path}")

# temporary file directory
cdir = "/tmp"

# initialize model and manager
model = DeepPETEncoderGradCAM()
model_manager = DeepPETModelManager(model=model, odir=odir)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

try:
    # predict
    test_df = pd.read_csv(ds_path)
    # filter by test ids
    test_df = test_df[test_df['amypad_id'].isin(args.ids)]
    # QC check
    test_gen = DeepPETDataGenerator(
        args.data_dir,
        fpaths=args.ids,
    )
    test_ds = test_gen.create_dataset(cache_dir=cdir, mode="prediction")

    outputs = model_manager.predict(test_ds=test_ds)
    test_df["y_score"] = outputs
    test_df["y_hat"] = test_df["y_score"].apply(sigmoid)
    test_df.to_csv(os.path.join(odir, os.path.basename(ds_path)), index=False)

finally:
    # clear cache 
    pt_files = os.listdir(cdir)
    filtered_files = [file for file in pt_files if file.endswith(".pt")]
    print(f"removing: {filtered_files}")
    for file in filtered_files:
        path_to_file = os.path.join(cdir, file)
        os.remove(path_to_file)
    print("clean-up complete")
