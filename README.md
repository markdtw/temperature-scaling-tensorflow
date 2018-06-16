# Temperature Scaling tensorflow
Tensorflow implementation of [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).

What this repo can do:

- Train ResNet_v1_110
- Calibrate it's output on CIFAR-10/100
- Using ```temp_scaling``` function to calibrate any of your networks using tensorflow.

What this repo *cannot* do:

- Calculate ECE (Expected Calibration Error)

Official PyTorch implementation by @gpleiss [here](https://github.com/gpleiss/temperature_scaling).

## Prerequisites
- Python 3.5
- [NumPy](http://www.numpy.org/)
- [TensorFlow 1.8](https://www.tensorflow.org/)


## Data
- [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html)


## Preparation
- Create `data/` folder, download and extract the python version from CIFAR webpage.


## Train
First, train the model (ResNet 110 in this case) using default parameters:
```bash
python main.py
```
Check out tunable hyper-parameters:
```bash
python main.py --help
```

## Temperature Scaling
Then, do temperature scaling to calibrate your model on the validation set.
```bash
python temp_scaling.py
```
Use the ```temp_var``` returned by ```temp_scaling``` function with your models logits to get calibrated output.


## Notes
- ResNet_v1_110 is trained for 250 epochs with other default parameters introduced in the original ResNet paper.
- The identity shortcut in ResNet_v1_110 is replaced with projection shortcut, meaning there are two additional convolutional layers.
- Validation accuracy and test accuracy on CIFAR-100 are around 70%.
- Issues are welcome!


## Resources
- [The paper](https://arxiv.org/abs/1706.04599).
- [Official PyTorch Implementation](https://github.com/gpleiss/temperature_scaling)
