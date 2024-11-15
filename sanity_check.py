import argparse
import pandas as pd
from DeepPET.data import DeepPETDataGenerator
from monai.visualize.utils import matshow3d
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

# initialize parser
parser = argparse.ArgumentParser(description='DeepPET model testing')
parser.add_argument('data_dir',
                        help='path to save/find extracted and downloaded scans')  
parser.add_argument('--odir', nargs='?',
                        help='path to odir')  # assumes csv file is within data_dir if unspecified
parser.add_argument('--metadata', nargs='?',
                        help='path to metadata')  # assumes csv file is within data_dir if unspecified
parser.add_argument('--ids', nargs='*',
                        help='list of IDs to include for training. Example usage: --ids 1 2 3 ')      # default value is None
args = parser.parse_args()

# parse arguments 
test_df = pd.read_csv(args.metadata)
odir = args.odir
fpaths = [os.path.join(args.data_dir, f) for f in test_df['img_path'].values.flatten()] 

# QC check
test_gen = DeepPETDataGenerator(
    fpaths=fpaths,
)
# preprocess check
processed_imgs = test_gen.preprocess_for_visualization(test_gen.fpaths)

# visualise img
matshow3d(processed_imgs[0], frame_dim=-1, show=True, cmap='gray')

# save img
img_np = processed_imgs[0]

if os.path.exists(odir):
    shutil.rmtree(odir)
os.makedirs(odir, exist_ok=True)

for k in np.arange(img_np.shape[0]):
    plt.imshow(img_np[k, :, :], cmap="gray")
    plt.savefig(os.path.join(odir, f"sagittal{k}.png"))
    plt.close()

for k in np.arange(img_np.shape[1]):
    plt.imshow(img_np[:, k, :], cmap="gray")
    plt.savefig(os.path.join(odir, f"coronal_{k}.png"))
    plt.close()

for k in np.arange(img_np.shape[2]):
    plt.imshow(img_np[:, :, k], cmap="gray")
    plt.savefig(os.path.join(odir, f"axial_{k}.png"))
    plt.close()

