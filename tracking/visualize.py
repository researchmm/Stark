import os
import sys
import argparse



prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation.data import SequenceList, Sequence
from lib.test.utils.load_text import load_text
from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker
import numpy as np
import shutil

class TrackDataset():

    def __init__(self, dir_name):

        self.sequence_list = os.listdir(dir_name)
        self.base_path = dir_name

    def get_sequence_list(self):
        has_folders = 0
        for f in self.sequence_list:
            if os.path.isdir(os.path.join(self.base_path, f)):
                has_folders = 1
                break
        if has_folders == 1:
            return SequenceList([self._construct_sequence(s) for s in self.sequence_list])
        return SequenceList([self._construct_sequence('')])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}'.format(self.base_path, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
        target_visible = np.loadtxt('{}/{}/absence.label'.format(self.base_path, sequence_name))
        return Sequence(sequence_name if sequence_name != '' else os.path.split(self.base_path)[1], frames_list, 'got10k',
                        ground_truth_rect.reshape(-1, 4), target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

def run_tracker(tracker_name, tracker_param, dir_name, run_id=None, debug=0, threads=0,
                num_gpus=8):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = TrackDataset(dir_name).get_sequence_list()
    res_dir = os.path.join(os.path.split(dir_name)[0], 'results')

    if os.path.exists(res_dir):
        shutil.rmtree(res_dir, ignore_errors=True)
    os.mkdir(res_dir)
    print(dataset)

    trackers = [Tracker(tracker_name, tracker_param, 'got10k_test', run_id)]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus, visualize=True, dir_name=res_dir)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('dir_name', type=str)
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=8)

    args = parser.parse_args()

    run_tracker(args.tracker_name, args.tracker_param,  args.dir_name, args.runid, args.debug,
                args.threads, num_gpus=args.num_gpus)
'''
tracker_name may be one of the following: stark_s, stark_st, stark_lightning_X_trt
tracker_param: baseline; other possible options are baseline_R101, baseline_got10k_only - were not trained yet, use baseline
'''

if __name__ == '__main__':
    main()
