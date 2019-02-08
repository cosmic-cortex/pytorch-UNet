# pytorch-UNet

A tunable implementation of [U-Net](https://arxiv.org/abs/1505.04597) in PyTorch.

- [U-Net quickstart](#quickstart)
- [Customizing the network](#customizing)
- [How to use](#usage)
- [Experiments with U-Net](#experiments)

## About U-Net<a name="unet"></a>

<img src="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png" width="500" align="middle"/>  
U-Net is a powerful encoder-decoder CNN architecture for semantic segmentation, developed by Olaf Ronneberger, 
Philipp Fischer and Thomas Brox. It has won several competitions, for example the ISBI Cell Tracking Challenge 2015 or
the Kaggle Data Science Bowl 2018.   

An example image from the Kaggle Data Science Bowl 2018:
<p float="left" align="middle">
  <img src="docs/img/tissue_original.png" width="256" />
  <figcaption>original image</figcaption>
  <img src="docs/img/tissue_segmented.png" width="256" />
  <figcaption>U-Net segmentation</figcaption> 
</p>

This repository was created to 
1. provide a reference implementation of 2D and 3D U-Net in PyTorch,
2. allow fast prototyping and hyperparameter tuning by providing an easily parametrizable model. 

In essence, the U-Net is built up using encoder and decoder blocks, each of them consisting of convolutional
and pooling layers. With this implementation, you can build your U-Net using the `First`, `Encoder`, `Center`,
`Decoder` and `Last` blocks, controlling the complexity and the number of these blocks. 
(Because the first, last and the middle of these blocks are somewhat special, they require their own class.)  

**WARNING!** The 3D U-Net implementation is currently untested!  

## U-Net quickstart<a name="quickstart"></a>

The simplest way to use the implemented U-Net is with the provided `train.py` and `predict.py` scripts.  
For training, `train.py` should be used, where the required arguments are
- `--train_dataset`: path to the training dataset which should be structured like
```
images_folder
   |-- images
       |-- img001.png
       |-- img002.png
       |-- ...
   |-- masks
       |-- img001.png
       |-- img002.png
       |-- ...
```
- `--checkpoint_path`: path to the folder where you wish to save the results (the trained model, predictions
for images in the validation set and log of losses and metrics during training).

Optional arguments:
- `--val_dataset`: path to the validation dataset, having the same structure as the training
dataset indicated above. Defaults to None.
- `--device`: the device where you wish to perform training and inference. Possible values are 'cpu',
'cuda:0', 'cuda:1', etc. Defaults for 'cpu'.
- `--in_channels`: the number of channels in your images. Defaults to 3.
- `--out_channels`: the number of classes in your image. Defaults to 2.
- `--depth`: controls the depth of the network. Defaults to 5. (For detailed explanation, see
[Customizing the network](#customizing).)
- `--width`: the complexity of each block. More width = more filters learned per layer. Defaults to 32. (For detailed explanation, see
[Customizing the network](#customizing).)
- `--epochs`: number of epochs during training. Defaults to 100.
- `--batch_size`: the size of the minibatch during each training loop. Defaults to 1. You should tune this to
completely fill the memory of your GPU.
- `--save_freq`: the frequency of saving the model and predictions. Defaults to 0, which results in not saving
the model at all, only the logs.
- `--save_model`: 1 if you want to save the model at every `save_freq` epochs and 0 if you dont. Defaults to 0. 
- `--model_name`: name of the model. Defaults to `model`. (Useful for parameter tuning.)
- `--learning_rate`: the learning rate for training. Defaults to 1e-3.
- `--crop`: integer describing the size of random crops taken from training and validation images.
Defaults to None, which results in no cropping. For example, if set to 256, the model is trained and
valdiated on (256, 256) sized random crops. 

## Customizing the network<a name="customizing"></a>

## Experiments with U-Net<a name="experiments"></a>
