# Deep Learning For Dummies

- [Models](#models)
  - [Deep Neural Network](#deep-neural-networks)
    - [Model 1](#model-1-dnn)
  - [Convolutional Neural Network](#convolutional-neural-networks)
    - [Model 1](#model-1-cnn)
  - [VGG16 Architecture](#vgg-16-architecture)
    - [Model 1](#model-1-vgg16)
    - [Model 2](#model-2-vgg16)
    - [Model 3](#model-3-vgg16)
  - [UNET Architecture](#unet-architecture)
    - [Model 1](#model-1-unet)
    - [Model 2](#model-2-unet)
    - [Model 3](#model-3-unet)
- [Courses](#courses)
- [Resources](#resources)
  - [TensorFlow](#tensorflow)
  - [Keras](#keras)
  - [PyTorch](#pytorch)
  - [PyTorch Lightning](#pytorch-lightning)

# Models
## Deep Neural Networks

![DNN](https://github.com/Antonio-Villarreal/DeepLearningModels/blob/main/Resources/Neural%20Network%20Visual.jpeg)

### Model 1 (DNN)

- [Deep Neural Network](https://colab.research.google.com/drive/1ydb9ww3bMfoFe74xJAxrBftPDytn42X2?usp=sharing) created using TensorFlow/Keras and the MNIST dataset. 60,000 images each 28 pixels x 28 pixels with 10 possible classifications.
  - Layers
    - Input Layer: 784 Nodes
    - Hidden Layer 1: 512 Nodes
    - Hidden Layer 2: 512 Nodes
    - Output Layer: 10 Nodes
  - Details
    - Optimizer: adam
    - Loss Function: categorical_crossentropy
  - Accuracy
    - 98.12 %

## Convolutional Neural Networks

- [Convolutional Neural Networks (CNN) — Architecture Explained](https://medium.com/@draj0718/convolutional-neural-networks-cnn-architectures-explained-716fb197b243)
- [VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/pdf/1409.1556.pdf)

![CNN](https://github.com/Antonio-Villarreal/DeepLearningModels/blob/main/Resources/Convolutional%20Neural%20Network.png)

### Model 1 (CNN)

- [Convolutional Neural Network](https://colab.research.google.com/drive/1DcrntEMfznsbIOT0yzZbGDTF9UGslY81#scrollTo=bP-s7oEidBri) created using TensorFlow/Keras and the MNIST dataset. The weights are preloaded from a file due to the time taken for each epoch.
  - Layers
    - 28x28x1
      - Convolving: 5x5x32
    - 24x24x32
      - Pooling 2x2
    - 12x12x32
      - Convolving: 5x5x64
    - 8x8x64
      - Pooling: 2x2
    - 4x4x64 (Input)
    - Hidden Layer: 1024 Nodes
    - Output Layer: 10 Nodes
  - Details
    - Optimizer: adam
    - Loss Function: categorical_crossentropy
  - Accuracy
    - 99.30 %

## VGG-16 Architecture

![CNN2](https://github.com/Antonio-Villarreal/DeepLearningStuff/blob/main/Resources/VGG16.png)

### Model 1 (VGG16)

- [VGG-16 Pretrained Predictor Model with ImageNet](https://colab.research.google.com/drive/1Q-PJClS1XzEHucVvsiO1kf7DgMXIWakY?usp=sharing) created using Keras with an imported version of VGG-16 and imported weights. It trains the model with the ImageNet dataset and enables you to pass an image into the model. It tries to predict what the image is and classify it based on the [categories](https://github.com/Antonio-Villarreal/Deep-Learning-For-Dummies/blob/main/Resources/Neural%20Networks%20and%20Convolutional%20Neural%20Networks/Ch07/07_02/data/synset_words.txt) offered by ImageNet.

### Model 2 (VGG16)

- [VGG16 Adaptations](https://colab.research.google.com/drive/1FT6ZLGHZ_m6JP31eYN6RBDZHkAKxMnW1?usp=sharing) uses the CIFAR10 dataset. There are multiple models that are meant to mimic VGG16, but are slightly adapted due to the image sizes being 32x32 versus 244x244. I experiement with Convolutional Layers, Batch Normalization, Dropout, and Batch Size while trying to beat the imported VGG16 model.

<p float="center">
  <img src="https://github.com/Antonio-Villarreal/Deep-Learning-For-Dummies/blob/main/Resources/Screenshot_20230126_110438.png" />
  <img src="https://github.com/Antonio-Villarreal/Deep-Learning-For-Dummies/blob/main/Resources/Screenshot_20230127_111359.png" />
</p>

### Model 3 (VGG16)

- [VGG16 from Scratch](https://colab.research.google.com/drive/1gFEMRUpINFR2Y81p9l0aD-4FfnBhEQzR?usp=sharing) uses PyTorch and the CIFAR10 Dataset. This is the best example of a Convolutional Neural Network implemented using PyTorch. Great reference for future models.

## UNET Architecture

- UNET Architectures are used for [Image Segmentation](https://blog.paperspace.com/unet-architecture-image-segmentation/)

![UNET](https://github.com/Antonio-Villarreal/Deep-Learning-For-Dummies/blob/main/Resources/UNET.png)

### Model 1 (UNET)

- [Oxford Pets](https://colab.research.google.com/drive/1JtiTqwKPBe32qh8M7xIPlP4ZMS9MxLqM?usp=sharing) is a dataset with numerous types of pets and usesd a UNET architecture in TensorFlow to segment images.

### Model 2 (UNET)

- [Carvana Image Segmentation](https://colab.research.google.com/drive/1JtiTqwKPBe32qh8M7xIPlP4ZMS9MxLqM?usp=sharing) is a dataset with numerous types of cars and is famous example for image segmentation. It uses PyTorch and an UNET architecture (no PyTorch Lightning).

### Model 3 (UNET)

- [Oxford Pets Part 2](https://colab.research.google.com/drive/1k62BBZ9O4z0f1fCb4UWsYCyJY7bd9dey?usp=sharing) is using PyTorch Lightning! Make sure to add Jacard and Dice Number for better metrics.

# Courses

- [Neural Networks and Convolutional Neural Networks Essential Training](https://www.linkedin.com/learning/neural-networks-and-convolutional-neural-networks-essential-training/welcome?autoplay=true&resume=false&u=41282748)
  - Introduction to Neural Networks (Fundamentals, XOR Challenege)
  - Components of Neural Networks (Activation Functions, Visualization)
  - Neural Networks Implementation in Keras
  - Convolutional Neural Networks + Keras (Convolutions, Zero Padding, Pooling)
  - Enhancements/Image Augmentation to Convolutional Neural Networks
  - ImageNet + VGG16
  - TensorFlow
  
- [TensorFlow: Working with Images](https://www.linkedin.com/learning/tensorflow-working-with-images/work-with-gray-and-color-images-using-transfer-learning-and-fine-tuning?u=41282748)
  - Neural Networks and Images (TensorFlow Hub, Hyperparameters)
  - Transfer Learning
  - Monitoring the Training Process (ModelCheckpoint, EarlyStopping, Tensorboard
  - TensorFlow
  
- [TensorFlow: Neural Networks and Working with Tables](https://www.linkedin.com/learning/tensorflow-neural-networks-and-working-with-tables/using-tensorflow-for-neural-networks-and-tables?u=41282748)
  - Fashion MNIST and Neural Networks
  - Loss, Gradient Descent, and Optimizers
  - Tabular Data
  - TensorFlow
  
- [PyTorch Essential Training: Deep Learning](https://www.linkedin.com/learning/pytorch-essential-training-deep-learning/welcome?autoplay=true&u=41282748)
  - Fashion MNIST and Neural Networks
  - Classes and Tensors
  - Loss, Autograd, and Optimizers
  - CPU/GPU
  - PyTorch
  
- [Transfer Learning for Images Using PyTorch: Essential Training](https://www.linkedin.com/learning/transfer-learning-for-images-using-pytorch-essential-training/welcome?autoplay=true&u=41282748)
  - [Notes](https://colab.research.google.com/drive/1Wee_c7EDG5ItKTHQhwetMAGBLWqYltSu?usp=sharing)
  - CIFAR10, Pretrained VGG16, PyTorch basics
  
- [DeepLizard Videos](https://www.youtube.com/watch?v=v5cngxo4mIg&list=PLZbbT5o_s2xrfNyHZsM6ufI0iZENK9xgG) 
  - [Notes](https://colab.research.google.com/drive/1XG4lOUjl5sMySK4TxqWEUaGmhnOa85uA?usp=sharing)
  - PyTorch basics, classes, and layers
  
- [PyTorch Image Segmentation Tutorial with U-NET: everything from scratch baby](https://www.youtube.com/watch?v=IHq1t7NxS8k)
  - PyTorch, UNET, Image Segmentation
  
- [Getting Started with PyTorch Lightning for Deep Learning](https://www.youtube.com/watch?v=aPVGQqrZaaU) (INCOMPLETE)
  - PyTorch Lightning, Deep Learning

- [Learn PyTorch for deep learning in a day. Literally.](https://youtu.be/Z_ikDlimN6A) (INCOMPLETE)
  - Everything PyTorch - Fundamentals, Workflow, Neural Network Classification, Computer Vision, Custom Datasets 
  
# Resources
- [Deep Learning State of the Art (2020)](https://www.youtube.com/watch?v=0VH1Lim8gL8&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf)
- [Complete MIT Deep Learning Course](https://deeplearning.mit.edu/)
- [Deep Learning MIT Book](https://www.deeplearningbook.org/)
- [OneNote](https://uflorida-my.sharepoint.com/:o:/g/personal/a_villarreal1_ufl_edu/EqN_9uO1-XNMmhl5iqskOEYBs22S03ytZV7OD-RiHwK_4g?e=oHywni)

## TensorFlow
- [Documentation](https://www.tensorflow.org/api_docs)
- [TensorFlow Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.82501&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
- [TensorFlow Hub](https://www.tensorflow.org/hub)

## Keras
- [Documentation](https://keras.io/)

## PyTorch
- [Documentation](https://pytorch.org/docs/stable/index.html)
- [Learn PyTorch](https://www.learnpytorch.io/)
- [Training A Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

## PyTorch Lightning
- [Tutorials](https://www.pytorchlightning.ai/tutorials)
- [From PyTorch to PyTorch Lightning](https://www.youtube.com/watch?v=DbESHcCoWbM&list=PLaMu-SDt_RB5hhJKZC5a6HPdlDTawUT3r&index=2)
- [Examples of PyTorch -> PyTorch Lightning](https://colab.research.google.com/drive/1-mLHUN4JnKC63kVUZuHRkp_aLWarkB0r?usp=sharing)
