import argparse
import os
import pandas as pd
from DeepPET.data import DeepPETDataGenerator
from DeepPET.architecture import DeepPETEncoderGradCAM
from DeepPET.model import DeepPETModelManager
from DeepPET.utils import sigmoid

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
ids = args.ids
data_dir = args.data_dir

# fpaths = []
# for amypad_id in args.ids:
    # filepaths = glob.glob(os.path.join(data_dir, '**', f"*{amypad_id.replace('-','')}*T1w_pet.nii.gz"), recursive=True)
    # fpaths.append(filepaths[0]) 

# temporary file directory
cdir = "tmp"

# initialize model and manager
model = DeepPETEncoderGradCAM()
model_manager = DeepPETModelManager(model=model, odir=odir, checkpoint=os.path.join(odir, 'model.pth'))

try:
    # predict
    test_df = pd.read_csv(args.metadata)
    fpaths = [os.path.join(data_dir, f) for f in test_df['img_path'].values.flatten()] 

    # QC check
    test_gen = DeepPETDataGenerator(
        fpaths=fpaths
    )
    test_ds = test_gen.create_dataset(cache_dir=cdir, mode="prediction")

    outputs = model_manager.predict(test_ds=test_ds)
    test_df["y_score"] = outputs
    test_df["y_prob"] = test_df["y_score"].apply(sigmoid)
    test_df["y_pred"] = (test_df["y_prob"]>0.5).astype(int)
    test_df.to_csv('predictions.csv', index=False)


finally:
    # clear cache 
    pt_files = os.listdir(cdir)
    filtered_files = [file for file in pt_files if file.endswith(".pt")]
    print(f"removing: {filtered_files}")
    for file in filtered_files:
        path_to_file = os.path.join(cdir, file)
        os.remove(path_to_file)
    print("clean-up complete")
