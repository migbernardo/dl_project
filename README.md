Repository of deep learning course project 

Useful links

1. [EuroSat dataset](https://github.com/phelber/EuroSAT)

2. [EuroSat paper](https://ieeexplore.ieee.org/document/8519248) 

3. [Deep learning with small data](https://arxiv.org/pdf/2003.12843.pdf)

4. [Keras - Image data preprocessing](https://keras.io/api/preprocessing/image/)

5. [Keras - Model training APIs](https://keras.io/api/models/model_training_apis/) 

6. [Keras - Optimizers](https://keras.io/api/optimizers/)

7. [Keras - Transfer learning](https://keras.io/guides/transfer_learning/) 

8. [Keras - Learning rate schedules](https://keras.io/api/optimizers/learning_rate_schedules/)

9. [Tensorflow - ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) 

10. [Data augmentation tutorial](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)

11. [Keras models](https://keras.io/api/applications/)

11. [Keras - ResNet](https://keras.io/api/applications/resnet/) 

12. [Keras - Model saving APIs](https://keras.io/api/models/model_saving_apis/#savemodel-function)

13. [Keras - BatchNormalization](https://keras.io/api/layers/normalization_layers/batch_normalization/) 

14. [Batch Normalization](https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/)

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

Shallow CNN 
- train_samples = 150-200 
- dropout = 0.7
- data augmentation 
- reduce number of filters

Baselines 
- Shallow CNN: 0.9119 / 0.9033 (val / test) / 860.15 seconds
- ResNet50: 0.9576 / 0.9556 (val / test) / 2504.20 seconds

Model 1 
- all samples
- 32 filters 
- 1 Dropout layer (rate=0.5)
- 120 epochs 
- 500 steps per epoch
- augmentation 
- 0.8761 / 0.8704 (val / test)
- 905.55 seconds

Model 2 
- all samples
- 2 x stack convolution layers 
- 32 and 64 filters
- 1 Dropout layer (rate=0.5)
- 100 epochs 
- 500 steps per epoch
- augmentation 
- 0.9165 / 0.9076 (val / test)
- 866.20 seconds

Model 3 
- all samples
- 2 x stack convolution layers 
- 32 and 64 filters
- 3 x dropout layers (rate=0.25 and 0.5)
- 100 epochs 
- 500 steps per epoch
- augmentation 
- 0.9274 / 0.9124 (val / test)
- 899.37 seconds

Model 4
- all samples
- 2 x stack convolution layers 
- 32 and 64 filters
- 3 x BatchNormalization layers
- 100 epochs 
- 500 steps per epoch
- augmentation 
- 0.8796 / 0.8752 (val / test)
- 936.91 seconds 

Model 5
- all samples
- 2 x stack convolution layers 
- 32 and 64 filters
- 3 x dropout layers (rate=0.25 and 0.5)
- Dense layer with 64 units
- 100 epochs 
- 500 steps per epoch
- augmentation 
- 0.9107 / 0.8952 (val / test)
- 888.19 seconds 

Model 6
- 4860 samples
- 2 x stack convolution layers 
- 32 and 64 filters
- 3 x dropout layers (rate=0.25 and 0.5)
- 100 epochs 
- 50 steps per epoch
- augmentation 
- 0.7228 / 0.7083 (val / test)
- 367.62 seconds

Model 7
- 4860 samples
- 2 x stack convolution layers 
- 32 and 64 filters
- 3 x dropout layers (rate=0.5 and 0.7)
- 100 epochs 
- 50 steps per epoch
- augmentation 
- 0.6678 / 0.6383 (val / test)
- 367.78 seconds 

Model 8
- 4860 samples
- 2 x stack convolution layers 
- 32 and 64 filters
- 3 x dropout layers (rate=0.25 and 0.5)
- 100 epochs 
- 200 steps per epoch
- augmentation 
- 0.8380 / 0.8450 (val / test)
- 539.45 seconds 

Breaking point of model 8 

- 6480 samples 
- 0.8807 / 0.8696 (val / test) 
- 537.94 seconds 

- 3240 samples 
- 0.8509 / 0.8474 (val / test) 
- 536.83 seconds 

- 1620 samples 
- 0.7928 / 0.7717 (val / test) 
- 419.46 seconds

Model 9
- 4860 samples
- 2 x stack convolution layers 
- 32 and 64 filters
- 3 x dropout layers (rate=0.25 and 0.5)
- batch size 32
- 100 epochs 
- 100 steps per epoch
- augmentation 
- 0.8309 / 0.8181 (val / test)
- 498.88 seconds

Model 10 
- 4860 samples
- 2 x stack convolution layers 
- 64 and 128 filters
- 3 x dropout layers (rate=0.25 and 0.5)
- 100 epochs 
- 200 steps per epoch
- augmentation 
- 0.8604 / 0.8587 (val / test)
- 778.85 seconds 

