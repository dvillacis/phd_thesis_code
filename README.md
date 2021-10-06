# PhD Thesis Code
This code supports the PhD thesis "Total Variation Bilevel Learning: Optimality Conditions and Numerical Solution" that can be found [here]().

## Installation
```bash
$ conda create -n bilevel
$ conda activate bilevel
$ conda install numpy pillow scikit-image
$ pip install pylops pyproximal
```

## Learning Data Parameter
### Scalar
```bash
$ conda activate bilevel
$ python data_learning.py datasets/cameraman_128_5/filelist.txt -t scalar -i 15
```
### Patch
```bash
$ conda activate bilevel
$ python data_learning.py datasets/cameraman_128_5/filelist.txt -t patch -ps 4 -i 15
```

## Learning Regularization Parameter
### Scalar
```bash
$ conda activate bilevel
$ python regularization_learning.py datasets/cameraman_128_5/filelist.txt -t scalar -i 0.1
```
### Patch
```bash
$ conda activate bilevel
$ python regularization_learning.py datasets/cameraman_128_5/filelist.txt -t patch -ps 4 -i 0.1
```

## Optional Parameters
1. -o/--output: Directory where to store the results
2. -v/--verbose: Print evolution of iterations of the bilevel solver
3. -i/--init: Initial value for the parameter
4. -ps/--patch_size: Patch size of the parameter

### Example
```bash
$ python data_learning.py datasets/cameraman_128_5/filelist.txt -t scalar -i 10 -o output/cameraman_128_5/10/ -v
```