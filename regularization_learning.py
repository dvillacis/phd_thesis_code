import argparse
import warnings
import os
import sys
import numpy as np

warnings.filterwarnings('ignore')

from bilearning.Learning.optimization import find_optimal_reg_scalar, find_optimal_reg_patch
from bilearning.Report.output_report import write_scalar_report, write_patch_report

parser = argparse.ArgumentParser(prog='bilevel_regularization_learning',description='Bilevel Regularization Parameter Learning',epilog='Enjoy!')

parser.add_argument('Dataset',metavar='dataset_file',type=str,help='Path to dataset file')
parser.add_argument('-t','--type',metavar='parameter_type',type=str,help='Parameter type scalar/patch')
parser.add_argument('-ps','--patch_size',metavar='patch_size',type=int,help='Patch parameter size')
parser.add_argument('-i','--init',metavar='parameter_init',type=float,help='Parameter initial value scalar/patch')
parser.add_argument('-o','--output',metavar='output_report_dir',type=str,help='Directory for storing output report')
parser.add_argument('-v','--verbose',action='store_true',help='Verbosity level on optimization')
parser.add_argument('-g','--gamma',metavar='smoothing_parameter', type=float, help='Smoothing parameter for the TV norm')
parser.add_argument('-tr','--threshold_radius',metavar='threshold_radius',type=float,help='Threshold radius for model switching')

args = parser.parse_args()

dataset_file = args.Dataset
parameter_type = 'scalar'
paramater_initial_value = 10.0
patch_size = 4
show = False
gamma = 100000
thr = 1e-4
report_dir = None
if args.type != None:
    parameter_type = args.type
if args.init != None:
    paramater_initial_value = args.init
if args.verbose == True:
    show=True
if args.output != None:
    report_dir = args.output
if args.patch_size != None:
    patch_size = args.patch_size
if args.gamma != None:
    gamma = args.gamma
if args.threshold_radius != None:
    thr = args.threshold_radius

if not os.path.isfile(dataset_file):
    raise ValueError(f'Dataset file: {dataset_file} not found...')

if parameter_type == 'scalar':
    optimal,optimal_ds = find_optimal_reg_scalar(dataset_file,paramater_initial_value,show=show,gamma=gamma,threshold_radius=thr)
    if optimal.status == 1:
        print(f'Optimal parameter found:\n{optimal.x}')
    else:
        print(f'Learning finished with status: {optimal.message}')
    if report_dir != None:
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
            print(f'Directory {report_dir} created successfully...')
        write_scalar_report(report_dir,optimal,optimal_ds)
else:
    paramater_initial_value = paramater_initial_value * np.ones((patch_size,patch_size))
    optimal,optimal_ds = find_optimal_reg_patch(dataset_file,paramater_initial_value,show=show)
    if optimal.status == 1:
        print(f'Optimal parameter found:\n{optimal.x}')
    else:
        print(f'Learning finished with status: {optimal.message}')
    if report_dir != None:
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
            print(f'Directory {report_dir} created successfully...')
        write_patch_report(report_dir,optimal,optimal_ds)


