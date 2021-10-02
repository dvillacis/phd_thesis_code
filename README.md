# PhD Thesis Code
This code supports the PhD thesis "Total Variation Bilevel Learning: Optimality Conditions and Numerical Solution" that can be found [here]().

## Installation
```bash
$ conda create -n bilevel
$ conda activate bilevel
$ conda install numpy pillow scikit-image pylops
$ pip install pyproximal
```

## Learning Data Parameter
### Scalar
```bash
$ conda activate bilevel
$ python data_learning.py datasets/cameraman.data -t scalar
```
### Patch
```bash
$ conda activate bilevel
$ python data_learning.py datasets/cameraman.data -t patch -s 4
```

## Learning Regularization Parameter
### Scalar
```bash
$ conda activate bilevel
$ python regularization_learning.py datasets/cameraman.data -t scalar
```
### Patch
```bash
$ conda activate bilevel
$ python regularization_learning.py datasets/cameraman.data -t patch -s 4
```