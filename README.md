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
$ python data_learning.py datasets/cameraman.ds -t scalar -i 15
```
### Patch
```bash
$ conda activate bilevel
$ python data_learning.py datasets/cameraman.ds -t patch -s 4 -i 15
```

## Learning Regularization Parameter
### Scalar
```bash
$ conda activate bilevel
$ python regularization_learning.py datasets/cameraman.ds -t scalar -i 0.1
```
### Patch
```bash
$ conda activate bilevel
$ python regularization_learning.py datasets/cameraman.ds -t patch -s 4 -i 0.1
```