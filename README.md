Repository of deep learning course project 

Useful links

1. [EuroSat dataset](https://github.com/phelber/EuroSAT)

2. [EuroSat paper](https://ieeexplore.ieee.org/document/8519248) 

3. [Deep learning with small data](https://arxiv.org/pdf/2003.12843.pdf)

4. [Keras - Image data preprocessing](https://keras.io/api/preprocessing/image/)

5. [Keras - Model training APIs](https://keras.io/api/models/model_training_apis/)

6. [Keras - Transfer learning](https://keras.io/guides/transfer_learning/) 

7. [Keras - ResNet](https://keras.io/api/applications/resnet/)

Networks to train

Shallow CNN 
- 3 layers composed of a 3x3 convolutional layer (stride=1, activation=relu) and a subsquent 4x4 max pooling layer (stride=2); followed by a fully connected layer 
- initial learning rate = 0.001
- epochs = 120 
- batch size = 16 
- decay = 1e-6 
- momentum (nesterov) = 0.9 
- optimizer = sgd 
- loss = categorial crossentropy 
- data augmentation - horizontal flipping, shearing (0.2 range), and random zooming (0.2 range)
- accuracy = 92.61%

ResNet-50 
- pretrained on ILSVRC-2012 dataset
- last layer learning rate = 0.01
- fine-tuning learning rate = 0.001 and 0.0001 
- accuracy = 98.57%

- Shallow CNN 
- train_samples = 150-200 
- dropout = 0.7
- data augmentation
