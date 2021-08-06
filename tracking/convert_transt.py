import _init_paths
import os
from lib.test.evaluation import get_dataset
import shutil

trackers = []
# dataset_name = 'uav'
dataset_name = 'nfs'


root_dir = "/data/sda/v-yanbi/iccv21/STARK_Latest/Stark"
base_dir = os.path.join(root_dir, "test/tracking_results/TransT_N2")
dataset = get_dataset(dataset_name)
for x in dataset:
    seq_name = x.name
    file_name = "%s.txt" % (seq_name.replace("nfs_", ""))
    file_path = os.path.join(base_dir, file_name)
    file_path_new = os.path.join(base_dir, "%s.txt" % seq_name)
    if os.path.exists(file_path):
        shutil.move(file_path, file_path_new)

