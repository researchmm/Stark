import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import Tracker, get_dataset, trackerlist

trackers = []


trackers.extend(trackerlist(name='stark_s', parameter_name='baseline',
                            run_ids=None, display_name='STARK-ST50'))

dataset = get_dataset('lasot')
# dataset = get_dataset('otb')
plot_results(trackers, dataset, 'LaSOT', merge_results=True, plot_types=('success'),
             skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, 'LaSOT', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
# print_per_sequence_results(trackers, dataset, report_name="debug")
