# Deep Learning For Dummies

- [Models](#models)
  - [Deep Neural Network (DNN)](#deep-neural-networks)
  - [Convolutional Neural Network (CNN)](#convolutional-neural-networks)
    - [VGG16 Architecture](#vgg-16-architecture-pt-1)
- [Courses](#courses)
- [Resources](#resources)
  - [TensorFlow](#tensorflow)
  - [Keras](#keras)
  - [PyTorch](#pytorch)

# Models
## Deep Neural Networks

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
  
![DNN](https://github.com/Antonio-Villarreal/DeepLearningModels/blob/main/Resources/Neural%20Network%20Visual.jpeg)

## Convolutional Neural Networks

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
![CNN](https://github.com/Antonio-Villarreal/DeepLearningModels/blob/main/Resources/Convolutional%20Neural%20Network.png)

### VGG-16 Architecture Pt 1

- [VGG-16 Pretrained Predictor Model with ImageNet](https://colab.research.google.com/drive/1Q-PJClS1XzEHucVvsiO1kf7DgMXIWakY?usp=sharing) created using Keras with an imported version of VGG-16 and imported weights. It trains the model with the ImageNet dataset and enables you to pass an image into the model. It tries to predict what the image is and classify it based on the [categories](https://github.com/Antonio-Villarreal/Deep-Learning-For-Dummies/blob/main/Resources/Neural%20Networks%20and%20Convolutional%20Neural%20Networks/Ch07/07_02/data/synset_words.txt) offered by ImageNet.

#### VGG-16 Model Summary AKA Neural Network Layers
```
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_3 (InputLayer)        [(None, 32, 32, 3)]       0         

     block1_conv1 (Conv2D)       (None, 32, 32, 64)        1792      

     block1_conv2 (Conv2D)       (None, 32, 32, 64)        36928     

     block1_pool (MaxPooling2D)  (None, 16, 16, 64)        0         

     block2_conv1 (Conv2D)       (None, 16, 16, 128)       73856     

     block2_conv2 (Conv2D)       (None, 16, 16, 128)       147584    

     block2_pool (MaxPooling2D)  (None, 8, 8, 128)         0         

     block3_conv1 (Conv2D)       (None, 8, 8, 256)         295168    

     block3_conv2 (Conv2D)       (None, 8, 8, 256)         590080    

     block3_conv3 (Conv2D)       (None, 8, 8, 256)         590080    

     block3_pool (MaxPooling2D)  (None, 4, 4, 256)         0         

     block4_conv1 (Conv2D)       (None, 4, 4, 512)         1180160   

     block4_conv2 (Conv2D)       (None, 4, 4, 512)         2359808   

     block4_conv3 (Conv2D)       (None, 4, 4, 512)         2359808   

     block4_pool (MaxPooling2D)  (None, 2, 2, 512)         0         

     block5_conv1 (Conv2D)       (None, 2, 2, 512)         2359808   

     block5_conv2 (Conv2D)       (None, 2, 2, 512)         2359808   

     block5_conv3 (Conv2D)       (None, 2, 2, 512)         2359808   

     block5_pool (MaxPooling2D)  (None, 1, 1, 512)         0         

    =================================================================
    Total params: 14,714,688
    Trainable params: 14,714,688
    Non-trainable params: 0
    _________________________________________________________________
```
![CNN2](https://github.com/Antonio-Villarreal/DeepLearningStuff/blob/main/Resources/VGG16.png)

### VGG-16 Architecture Pt 2

- [VGG-16 Fashion MNIST Dataset Model](https://colab.research.google.com/drive/1TX1tFMLwGj4R0llDvkNnawWkH3zjttdi?usp=sharing) created with Keras imported version of VGG-16. 1 Epoch takes approximately 1875 seconds which is ~32 minutes.

![CNN](https://github.com/Antonio-Villarreal/Deep-Learning-For-Dummies/blob/main/Resources/CNN.jpeg)


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
  - INCOMPLETE
  
- [Hands-On PyTorch Machine Learning](https://www.linkedin.com/learning/hands-on-pytorch-machine-learning/explore-the-capabilities-of-pytorch?autoplay=true&u=41282748)
  - INCOMPLETE
  
- [DeepLizard Videos](https://www.youtube.com/watch?v=v5cngxo4mIg&list=PLZbbT5o_s2xrfNyHZsM6ufI0iZENK9xgG) 
  - INCOMPLETE
  
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

## CNNs
- [Convolutional Neural Networks (CNN) — Architecture Explained](https://medium.com/@draj0718/convolutional-neural-networks-cnn-architectures-explained-716fb197b243)
- [VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/pdf/1409.1556.pdf)
