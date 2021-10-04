import argparse
import os
import sys

from Learning.optimization import find_optimal_data_scalar

parser = argparse.ArgumentParser(prog='bilevel_data_learning',description='Bilevel Data Parameter Learning',epilog='Enjoy!')

parser.add_argument('Dataset',metavar='dataset_file',type=str,help='Path to dataset file')
parser.add_argument('-t','--type',metavar='parameter_type',type=str,help='Parameter type scalar/patch')
parser.add_argument('-i','--init',metavar='parameter_init',type=float,help='Parameter initial value scalar/patch')

args = parser.parse_args()

dataset_file = args.Dataset
parameter_type = 'scalar'
paramater_initial_value = 10.0
if args.type != None:
    parameter_type = args.type
if args.init != None:
    paramater_initial_value = args.init

if not os.path.isfile(dataset_file):
    raise ValueError(f'Dataset file: {dataset_file} not found...')

if parameter_type == 'scalar':
    optimal = find_optimal_data_scalar(dataset_file,paramater_initial_value)
    print(f'Optimal parameter found:\n{optimal}')
else:
    print(f'Parameter of type {parameter_type} is not implemented yet...')


