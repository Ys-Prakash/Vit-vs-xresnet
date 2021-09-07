# ViT-Large vs Xresnet50 (using Adam and Ranger optimizers)
This repository is comprised of notebooks that contains code for testing Vision Transformer and Xresnet50, both of them pre-trained, on the ImageWoof dataset, using Adam and Ranger optimizers.

## Acknowledgements 

1. [The Walk with fatsai course on ImageWoof](https://walkwithfastai.com/ImageWoof)
2. [Ross Wightman's repository for vision transformer](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)
3. [Fastai](https://docs.fast.ai/)

## The Objective 

The objective of this project is to get a clear comparision between the performances of pre-trained Vision Transformer (here, ViT-Large) and pre-trained Xresnet50, when they are trained using different (here, Adam and Ranger) optimizers, on the ImageWoof dataset. This project is helpful to people who want to use state-of-the-art models, but have limited resources and are dependent on online environments such as Google Colab.


**NB :** For this project, ViT-Large was used as I (that is, Prakash Pandey) wanted to get the state-of-the-art model, and, also, because training ViT-Huge model threw 'CUDA Out of Memory' error even with batch size = 1, on Google Colab. So, I found ViT-Large to be the 'deepest' vision transformer that could be trained on Google Colab.

## The ImageWoof dataset 

The [ImageWoof](https://github.com/fastai/imagenette#imagewoof) dataset is a subset of 10 classes from Imagenet that aren't so easy to classify, since they're all dog breeds. The breeds are: Australian terrier, Border terrier, Samoyed, Beagle, Shih-Tzu, English foxhound, Rhodesian ridgeback, Dingo, Golden retriever, Old English sheepdog.

## The Models

### ViT_Large : 
The vision transformer, introduced [here](https://arxiv.org/pdf/2010.11929.pdf), has some variants, and ViT-Large is one of them. It comprises 24 layers and 307M parameters. For this project, I have used a pre-trained ViT-Large model.

### Xresnet50 :
This is a pre-trained Resnet50 model with some tricks based on [Bag of Tricks for Resnet ](https://arxiv.org/pdf/1812.01187.pdf) paper. There are few other tricks as well :

1. [Mish](https://arxiv.org/vc/arxiv/papers/1908/1908.08681v1.pdf) - A new activation function that has shown fantastic results
2. [Self-Attention](https://arxiv.org/pdf/1805.08318.pdf) - Bringing in ideas from GAN's into image classification 
3. [MaxBlurPool](https://arxiv.org/pdf/1904.11486.pdf) - Better generalization
4. Flatten + Anneal Scheduling - Mikhail Grankin
5. [Label Smoothing Cross Entropy](https://arxiv.org/pdf/1906.11567.pdf) - A threshold base (were you close) rather than yes or no

## The Optimizers 

### Adam :
[Adam](https://arxiv.org/abs/1412.6980v5) is a method for efficient stochastic optimization that only requires first-order gradients with little memory requirement. The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients; the name Adam is derived from adaptive moment estimation.

### Ranger :
Ranger is an optimizer based on two seperate papers :
1. [On the Varience of the Adaptive Learning rate and Beyond, RAdam](https://arxiv.org/pdf/1908.03265.pdf)
2. [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/pdf/1907.08610.pdf)

## The Result

### 1. Model based comparision :
#### A. Using Adam :
The pre-trained ViT-Large model achieved an accuracy of 81.29%, whereas, the pre-trained xresnet50 model achieved 34.69%; both of them on the ImageWoof dataset.

#### B. Using Ranger :
The pre-trained ViT-Large model achieved an accuracy of 27.28%, whereas, the pre-trained xresnet50 model achieved 43.72%; both of them on the ImageWoof dataset.

### 2. Optimizer based comparision :
#### A. Using pre-trained ViT-Large model :
The pre-trained ViT-Large model achieved an accuracy of 81.29% with Adam, whereas, 27.28% with Ranger; both of them on the ImageWoof dataset.

#### B. Using pre-trained xresnet50 model :
The pre-trained xresnet50 model achieved an accuracy of 34.69% with Adam, whereas, 43.72% with Ranger; both of them on the ImageWoof dataset.

#### Clearly, we see that the best combination of models and optimizers, used herein, is ViT-Large + Adam.
