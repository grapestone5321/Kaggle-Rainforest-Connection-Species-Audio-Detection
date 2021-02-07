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

## RFCX Bagging
https://www.kaggle.com/kneroma/rfcx-bagging

### paths = [

      #1"../input/rfcx-best-performing-public-kernels/kkiller_inference-tpu-rfcx-audio-detection-fast_0861.csv",
      #2"../input/rfcx-best-performing-public-kernels/submission_khoongweihao_0845.csv",
      #3"../input/rfcx-best-performing-public-kernels/submission_mekhdigakhramanian_0824.csv",

      sub_score = np.sum(scores*weights[:, None, None], 0)


### weights: default = np.array([0.6, 0.4])

#1, #2: 

      weights = np.array([0.5, 0.5])     LB 0.868   ver2
      weights = np.array([0.55, 0.45])   LB 0.869   ver7
      weights = np.array([0.6, 0.4])     LB 0.869   ver1   ---  default
      weights = np.array([0.65, 0.35])   LB 0.869   ver6
      weights = np.array([0.7, 0.3])     LB 0.868   ver3




weights = np.array([0.6, 0.4]):

     #1, #2     LB 0.869   ver1   ---  default
     #1, #3     LB 0.863   ver4
     #2, #3     LB 0.842   ver5






-------

## RFCX Custom training with TPU
https://www.kaggle.com/ashusma/rfcx-custom-training-with-tpu

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

### dropout: default = 0.35

LEARNING_RATE = 0.001:

      dropout = 0.30     LB 0.795   ver10     
      dropout = 0.35     LB 0.815   ver5     ---  default
      dropout = 0.40     LB 0.780   ver9

## EPOCHS: default = 25

      EPOCHS = 15    LB 0.787   ver13
      EPOCHS = 18    LB 0.818   ver15
      EPOCHS = 19    LB 0.804   ver17
      EPOCHS = 20    LB 0.824   ver12      ---  best
      EPOCHS = 21    LB 0.788   ver16
      EPOCHS = 22    LB 0.801   ver14
      EPOCHS = 25    LB 0.815   ver5       ---  default
      EPOCHS = 50    LB 0.777   ver11   
    
-------


