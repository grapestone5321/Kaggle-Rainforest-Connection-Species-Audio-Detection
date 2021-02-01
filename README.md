# Kaggle-Rainforest-Connection-Species-Audio-Detection
Kaggle-Rainforest-Connection-Species-Audio-Detection

-------

## End Date (Final Submission Deadline): 
February 17, 2021 11:59PM UTC

-------

## The task:
The task consists of predicting the species present in each test audio file. 

Some test audio files contain a single species while others contain multiple. 

The predictions are to be done at the audio file level, i.e., no start/end timestamps are required.


-------

## Paper:


### One weird trick for parallelizing convolutional neural networks: #AlexNet
https://arxiv.org/pdf/1404.5997.pdf

### VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION: #VGG
https://arxiv.org/pdf/1409.1556.pdf


### Deep Residual Learning for Image Recognition: #ResNet
https://arxiv.org/pdf/1512.03385.pdf


### Aggregated Residual Transformations for Deep Neural Networks: ResNeXt
https://arxiv.org/pdf/1611.05431.pdf


--------

## Progress
### LB Best score: 


-------

## RFCX Custom training with TPU

### N_FOLDS: default = 5

      N_FOLDS = 5        LB error   ver1
      N_FOLDS = 4        LB error   ver3
      N_FOLDS = 3        LB 0.796   ver2

### LEARNING_RATE: default = 0.0015

N_FOLDS = 3:

      LEARNING_RATE = 0.005     LB error   ver4
      LEARNING_RATE = 0.0015    LB 0.796   ver2   ---  default
      LEARNING_RATE = 0.001     LB 0.815   ver5
      LEARNING_RATE = 0.0005    LB 0.769   ver6
