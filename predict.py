import argparse
import os
import pandas as pd
from DeepPET.data import DeepPETDataGenerator
from DeepPET.architecture import DeepPETEncoderGradCAM
from DeepPET.model import DeepPETModelManager
from DeepPET.utils import sigmoid
from sklearn.metrics import accuracy_score, roc_auc_score

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
cdir = "tmp"

# initialize model and manager
model = DeepPETEncoderGradCAM()
model_manager = DeepPETModelManager(model=model, odir=odir, checkpoint=os.path.join(odir, 'model.pth'))

try:
    # predict
    test_df = pd.read_csv(ds_path)
    # filter by test ids and take first visit
    test_df = test_df[(test_df['amypad_id'].isin(args.ids)) & (test_df['visit_number']==0 ) ][['amypad_id', 'pet_vr_classification']]
    y_true = (test_df['pet_vr_classification']=="Positive").astype(int).to_numpy()

    # QC check
    test_gen = DeepPETDataGenerator(
        args.data_dir,
        ids=args.ids,
        labels=y_true
    )
    test_ds = test_gen.create_dataset(cache_dir=cdir, mode="prediction")

    outputs = model_manager.predict(test_ds=test_ds)
    test_df["y_raw"] = outputs
    test_df["y_score"] = test_df["y_raw"].apply(sigmoid)
    test_df["y_hat"] = (test_df["y_score"]>0.5).astype(int)
    test_df.to_csv('predictions.csv', index=False)

    # Generate performance summary
    acc = accuracy_score(y_true, test_df["y_hat"])
    # auroc = roc_auc_score(y_true, test_df["y_score"])
    print('Test acc:', acc)
    # print('Test auroc:', auroc)
    print(y_true)

finally:
    # clear cache 
    pt_files = os.listdir(cdir)
    filtered_files = [file for file in pt_files if file.endswith(".pt")]
    print(f"removing: {filtered_files}")
    for file in filtered_files:
        path_to_file = os.path.join(cdir, file)
        os.remove(path_to_file)
    print("clean-up complete")
