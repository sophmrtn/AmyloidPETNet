import argparse
import pandas as pd
from DeepPET.data import DeepPETDataGenerator
from monai.visualize.utils import matshow3d

# initialize parser
parser = argparse.ArgumentParser(description='DeepPET model testing')
parser.add_argument('data_dir',
                        help='path to save/find extracted and downloaded scans')  

parser.add_argument('--metadata', nargs='?',
                        help='path to metadata')  # assumes csv file is within data_dir if unspecified
parser.add_argument('--ids', nargs='*',
                        help='list of IDs to include for training. Example usage: --ids 1 2 3 ')      # default value is None
args = parser.parse_args()

# parse arguments 
ft_df = pd.read_csv(args.metadata)
# filter by ids
ft_df = ft_df[ft_df['amypad_id'].isin(args.ids)]

# If label column is a string convert to int
if type(ft_df['pet_vr_classification'].values[0]) is str:
    ft_df['pet_vr_classification'] = (ft_df['pet_vr_classification']=="Positive").astype(int)


# QC check
test_gen = DeepPETDataGenerator(
    args.data_dir,
    ids=args.ids,
    labels=ft_df['pet_vr_classification'].values,
)
# preprocess check
processed_imgs = test_gen.preprocess_for_visualization(test_gen.fpaths)

# visualise img
matshow3d(processed_imgs[0], frame_dim=-2, show=True, cmap='gray')
