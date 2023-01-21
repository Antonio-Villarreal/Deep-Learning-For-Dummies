# DeepLearningModels

## Deep Neural Network

- [Deep Neural Network](https://colab.research.google.com/drive/1ydb9ww3bMfoFe74xJAxrBftPDytn42X2?usp=sharing) created using TensorFlow/Keras and the MNIST dataset.
  -Layers
    - Input Layer: 784 Nodes
    - Hidden Layer 1: 512 Nodes
    - Hidden Layer 2: 512 Nodes
    - Output Layer: 10 Nodes
  -Details
    - Optimizer: adam
    - Loss Function: categorical_crossentropy
  - Accuracy
    - 98.12 %
  
![DNN](https://github.com/Antonio-Villarreal/DeepLearningModels/blob/main/Resources/Neural%20Network%20Visual.jpeg)

- [Convolutional Neural Network](https://colab.research.google.com/drive/1DcrntEMfznsbIOT0yzZbGDTF9UGslY81#scrollTo=bP-s7oEidBri) created using TensorFlow/Keras and the MNIST dataset. The weights are preloaded from a file due to the time taken for each epoch.
  -Layers
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
  -Details
    - Optimizer: adam
    - Loss Function: categorical_crossentropy
  - Accuracy
    - 99.30 %

![CNN](https://github.com/Antonio-Villarreal/DeepLearningModels/blob/main/Resources/Convolutional%20Neural%20Network.png)
