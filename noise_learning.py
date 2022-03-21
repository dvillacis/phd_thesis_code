import argparse
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from bilearning.Operators.patch import Patch
from bilearning.Learning.optimization import find_optimal_data_scalar, find_optimal_data_patch
from bilearning.Report.output_report import write_scalar_report, write_patch_report

parser = argparse.ArgumentParser(prog='bilevel_data_learning',description='Bilevel Data Parameter Learning',epilog='Enjoy!')

parser.add_argument('Dataset',metavar='dataset_file',type=str,help='Path to dataset file')
parser.add_argument('-t','--type',metavar='parameter_type',type=str,help='Parameter type scalar/patch')
parser.add_argument('-prows','--patch_rows',metavar='patch_init',type=int,help='Patch parameter rows')
parser.add_argument('-pdata','--patch_data',metavar='patch_data',type=float,nargs='+',help='Patch parameter data')
parser.add_argument('-i','--init',metavar='parameter_init',type=float,help='Parameter initial value scalar/patch')
parser.add_argument('-o','--output',metavar='output_report_dir',type=str,help='Directory for storing output report')
parser.add_argument('-v','--verbose',action='store_true',help='Verbosity level on optimization')

args = parser.parse_args()

dataset_file = args.Dataset
parameter_type = 'scalar'
paramater_initial_value = 10.0
patch_rows = 2
patch_data = np.array([[10.0,10.0],[10.0,10.0]])
show = False
report_dir = None
if args.type != None:
    parameter_type = args.type
if args.init != None:
    paramater_initial_value = args.init
if args.verbose == True:
    show=True
if args.output != None:
    report_dir = args.output
if args.patch_rows != None:
    patch_rows = args.patch_rows
if args.patch_data != None:
    patch_data = np.array(args.patch_data).reshape((patch_rows,len(args.patch_data)//patch_rows))

if not os.path.isfile(dataset_file):
    raise ValueError(f'Dataset file: {dataset_file} not found...')

if parameter_type == 'scalar':
    paramater_initial_value = Patch(np.array([paramater_initial_value]),1,1)
    optimal,optimal_ds = find_optimal_data_scalar(dataset_file,paramater_initial_value,show=show)
    if optimal.status == 1:
        print(f'Optimal parameter found:\n{optimal.x}')
    else:
        print(f'{optimal.message}')
    if report_dir != None:
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
            print(f'Directory {report_dir} created successfully...')
        write_scalar_report(report_dir,optimal,optimal_ds)
else:
    px,py = patch_data.shape
    paramater_initial_value = Patch(patch_data.ravel(),px,py)
    optimal,optimal_ds = find_optimal_data_patch(dataset_file,paramater_initial_value,show=show)
    if optimal.status == 1:
        print(f'Optimal parameter found:\n{optimal.x}')
    else:
        print(f'{optimal.message}')
    if report_dir != None:
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
            print(f'Directory {report_dir} created successfully...')
        write_patch_report(report_dir,optimal,optimal_ds)


