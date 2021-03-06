## Deep learning course project 

Repository for storing the course project for the NOVA IMS master course Deep Learning (academic year 2021/2022)

### Description

This project is about the development of a simple convolutional neural network that can be quickly trained and used for land use or cover classification from satellite images. The idea was inspired in the work of [phelber](https://github.com/phelber/EuroSAT) and done for learning purposes.

![image](https://user-images.githubusercontent.com/91217958/164242074-586da8c6-549a-4832-8c43-44803981c7ca.png)


### Repo's structure

```
dl_project
|          ┌── raw <------- contains the raw images saved in their respective class directories
|          |                after running the load.py script in src/data, which also creates 
|          |                the data directory
├── data ──┤
|          └── processed <- contains the processed images after running the preprocessing.py 
|                           script in src/data
│
├── content <-------------- contains the project's pdf report
│          
├── models <--------------- contains the confusion_matrix.py and class_activation_map.py scripts and 
|                           the saved models as .h5 files after running the respective scripts
|                           in src/model
|                 
│
├── notebooks <------------ contains all data exploration jupyter notebooks
│
|         ┌── data <------- contains all scripts related with data generation or transformation
├── src ──┤
|         └── model <------ contains all scripts related with model training and saving
│
├── requeriments.txt <----- list of all packages used in the project can be installed on the virtual 
|                           environment with the terminal command: "pip install -r requirements.txt"
|
└── README.md <------------ info file
```

### Useful links

1. [EuroSat paper](https://ieeexplore.ieee.org/document/8519248) 

2. [Deep learning with small data](https://arxiv.org/pdf/2003.12843.pdf)

3. [Keras - Image data preprocessing](https://keras.io/api/preprocessing/image/)

4. [Keras - Model training APIs](https://keras.io/api/models/model_training_apis/) 

5. [Keras - Optimizers](https://keras.io/api/optimizers/)

6. [Keras - Transfer learning](https://keras.io/guides/transfer_learning/) 

7. [Keras - Learning rate schedules](https://keras.io/api/optimizers/learning_rate_schedules/)

8. [Tensorflow - ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) 

9. [Data augmentation tutorial](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)

10. [Keras models](https://keras.io/api/applications/)

11. [Keras - ResNet](https://keras.io/api/applications/resnet/) 

12. [Keras - Model saving APIs](https://keras.io/api/models/model_saving_apis/#savemodel-function)

13. [Keras - BatchNormalization](https://keras.io/api/layers/normalization_layers/batch_normalization/) 

14. [Batch Normalization](https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/)

### Trained networks

Baselines 
- Shallow CNN (mb_initial_model.py): 0.9119 / 0.9033 (val / test) / 860.15 seconds
- ResNet50 (mb_restnet_model.py): 0.9576 / 0.9556 (val / test) / 2504.20 seconds

Model 1 (mb_lightnet_model_1.py)
- all samples
- 32 filters 
- 1 Dropout layer (rate=0.5)
- 120 epochs 
- 500 steps per epoch
- augmentation 
- 0.8761 / 0.8704 (val / test)
- 905.55 seconds

Model 2 (mb_lightnet_model_2.py)
- all samples
- 2 x stack convolution layers 
- 32 and 64 filters
- 1 Dropout layer (rate=0.5)
- 100 epochs 
- 500 steps per epoch
- augmentation 
- 0.9165 / 0.9076 (val / test)
- 866.20 seconds

Model 3 (mb_lightnet_model_3.py)
- all samples
- 2 x stack convolution layers 
- 32 and 64 filters
- 3 x dropout layers (rate=0.25 and 0.5)
- 100 epochs 
- 500 steps per epoch
- augmentation 
- 0.9274 / 0.9124 (val / test)
- 899.37 seconds

Model 4 (mb_lightnet_model_4.py)
- all samples
- 2 x stack convolution layers 
- 32 and 64 filters
- 3 x BatchNormalization layers
- 100 epochs 
- 500 steps per epoch
- augmentation 
- 0.8796 / 0.8752 (val / test)
- 936.91 seconds 

Model 5 (mb_lightnet_model_5.py)
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

Model 6 (mb_lightnet_model_3_smallds.py)
- 4860 samples
- 2 x stack convolution layers 
- 32 and 64 filters
- 3 x dropout layers (rate=0.25 and 0.5)
- 100 epochs 
- 50 steps per epoch
- augmentation 
- 0.7228 / 0.7083 (val / test)
- 367.62 seconds

Model 7 (mb_lightnet_model_3_smallds_2.py)
- 4860 samples
- 2 x stack convolution layers 
- 32 and 64 filters
- 3 x dropout layers (rate=0.5 and 0.7)
- 100 epochs 
- 50 steps per epoch
- augmentation 
- 0.6678 / 0.6383 (val / test)
- 367.78 seconds 

Model 8 (mb_lightnet_model_3_smallds_3.py)
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

Model 9 (mb_lightnet_model_3_smallds_4.py)
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

Model 10 (mb_lightnet_model_3_smallds_5.py)
- 4860 samples
- 2 x stack convolution layers 
- 64 and 128 filters
- 3 x dropout layers (rate=0.25 and 0.5)
- 100 epochs 
- 200 steps per epoch
- augmentation 
- 0.8604 / 0.8587 (val / test)
- 778.85 seconds 

Model 3 for class activation maps (mb_lightnet_model_3_gmp.py)
