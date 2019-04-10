# Co-occurrence-Nueral-Network

Co-occurrence layer(CoL) is a neural network layer which can extract the spatial and statistic features of the input.
(Co-occurrence Neural Network CVPR2019). 

## Installation
CoL was implemented in TensowFlow 1.4.1 and Python 2.7.12. 

The network examples require additional lybraries to work properly: 
* numpy 1.13.3
* tempfile
* time
* os
* matplotlib

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install libraries.

```bash
pip install Python 2.7.12
```
 
## Usage

```python
import co_layer

# co-occurrence layer
with tf.name_scope('conn'):
    CoL = co_layer.CoNN_layer(
                input,
                co_shape=[5, 5],
                co_initializer=None,
                w_shape=[3, 3, 3],
                w_initializer=None,
                name='co_layer')
```

CoL - the output of the layer 

input - the input to the layer

co_chape - the shape of the 2D co-occurrence matrix

co_initializer - the initializer of the co-occurrence matrix

w_shape  - the shape of 3D spatial filter 

w_initializer - the initializer of the spatial filter 

## Limitations 

- The number of input and output channels must be the same.
- The spatial term is a 3D filter.
- The size of co-occurrence matrix can be defined up to [10x10] in order to speed the network convergence. (In our experience, input statistics does not require co-occurrence matrix larger than  [7x7]).

## Toy Example

This folder contains the code that implements the Toy Example of using the CoL. Its performances are compared to the performances of other common layers.



