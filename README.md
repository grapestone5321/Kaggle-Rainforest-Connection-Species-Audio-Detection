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

### Data Description

In this competition, you are given audio files that include sounds from numerous species. 

Your task is, for each test audio file, to predict the probability that each of the given species is audible in the audio clip. 

While the training files contain both the species identification as well as the time the species was heard, the time localization is not part of the test predictions.

Note that the training data also includes false positive label occurrences to assist with training.

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
### LB Best score: 0.879

-------



## [AutoML] [Inference] Audio Detection - Soli 346f45
https://www.kaggle.com/hypnotu/automl-inference-audio-detection-soliset

### ked = pd.DataFrame

      'Kernel ID': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I','J'],
      'Symbol':    ['SoliSet', '[Inference] ResNest RFCX Audio Detection',  
                    'notebookba481ef16a', 
                    'All-in-one RFCX baseline for beginners', 
                    'RFCX: train resnet50 with TPU',  
                    'RFCX Resnet50 TPU', 
                    'ResNet34 More Augmentations+Mixup+TTA (Inference)', 
                    '[Inference][TPU] RFCX Audio Detection Fast++',
                    'RFX Bagging Different Weights',
                    'resnetwavenet'],
      'Score':     [ 0.589 , 0.594 , 0.613 , 0.748 , 0.793 , 0.824 , 0.845 , 0.861, 0.876, 0.877 ],
      'File Path':
      #A '../input/audio-detection-soliset-201/submission.csv', 
      #B '../input/inference-resnest-rfcx-audio-detection/submission.csv', 
      #C '../input/minimal-fastai-solution-score-0-61/submission.csv', 
      #D '../input/all-in-one-rfcx-baseline-for-beginners/submission.csv', 
      #E '../input/rfcx-train-resnet50-with-tpu/submission.csv', 
      #F '../input/rfcx-resnet50-tpu/submission.csv', 
      #G '../input/resnet34-more-augmentations-mixup-tta-inference/submission.csv', 
      #H '../input/inference-tpu-rfcx-audio-detection-fast/submission.csv',
      #I '../input/rfcx-bagging-with-different-weights-0-876-score/submission.csv',
      #J '../input/resnet-wavenet-my-best-single-model-ensemble/submission.csv'],      
      'Note': ['xgboost & cuml(https://rapids.ai)', 
               'torch & resnest50', 
               'fastai.vision & torchaudio', 
               'torch & resnest50', 
               'tensorflow & tf.keras.Sequential', 
               'tensorflow & tf.keras.Sequential', 
               'tensorflow & classification_models.keras', 
               'torch & resnest50', 
               'bagging','0.877']                                                  
 
 ### sub.to_csv(                  
 
      "submission.csv"     LB 0.777   ver11
      "submission4.csv"    LB 0.879   ver2

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

## RFCX Bagging
https://www.kaggle.com/kneroma/rfcx-bagging

### paths = [

      #1"../input/rfcx-best-performing-public-kernels/kkiller_inference-tpu-rfcx-audio-detection-fast_0861.csv",
      #2"../input/rfcx-best-performing-public-kernels/submission_khoongweihao_0845.csv",
      #3"../input/rfcx-best-performing-public-kernels/submission_mekhdigakhramanian_0824.csv",
      #4"/kaggle/input/rainforestconnectionemsamble1/rainforest-877.csv"
      #5"/kaggle/input/rainforest879/rainforest-879.csv"

      sub_score = np.sum(scores*weights[:, None, None], 0)


### weights: default = np.array([0.6, 0.4])

### #1, #2: 

       weights = np.array([0.5, 0.5])     LB 0.868   ver2
       weights = np.array([0.55, 0.45])   LB 0.869   ver7
       weights = np.array([0.6, 0.4])     LB 0.869   ver1   ---  default
       weights = np.array([0.65, 0.35])   LB 0.869   ver6
       weights = np.array([0.7, 0.3])     LB 0.868   ver3

### #1, #3

      weights = np.array([0.5, 0.5])      LB         ver
      weights = np.array([0.6, 0.4])      LB 0.863   ver4
      weights = np.array([0.7, 0.3])      LB         ver
     
    
### #2, #3  

      weights = np.array([0.5, 0.5])      LB         ver
      weights = np.array([0.6, 0.4])      LB 0.842   ver5
      weights = np.array([0.7, 0.3])      LB         ver


### #1, #4

      weights = np.array([0.1, 0.9])      LB 0.877   ver12
      weights = np.array([0.2, 0.8])      LB 0.877   ver11
      weights = np.array([0.3, 0.7])      LB 0.876   ver10
      weights = np.array([0.4, 0.6])      LB 0.873   ver9
      
 ### #4, #5     

      weights = np.array([0.05, 0.05])    LB 0.878   ver19
      weights = np.array([0.1, 0.9])      LB 0.879   ver18
      weights = np.array([0.2, 0.8])      LB 0.879   ver14
      weights = np.array([0.4, 0.6])      LB 0.879   ver13
      weights = np.array([0.6, 0.4])      LB 0.877   ver15
      weights = np.array([0.8, 0.2])      LB 0.877   ver16
      weights = np.array([0.9, 0.1])      LB 0.877   ver17
 
 
 ### #1, #5     

      weights = np.array([0.05, 0.05])    LB 0.869   ver23
      weights = np.array([0.1, 0.9])      LB 0.879   ver22
      weights = np.array([0.15, 0.85])    LB 0.879   ver29
      weights = np.array([0.2, 0.8])      LB 0.879   ver21
      weights = np.array([0.4, 0.6])      LB 0.875   ver20
      weights = np.array([0.6, 0.4])      LB         ver
      weights = np.array([0.8, 0.2])      LB 0.862   ver24
      weights = np.array([0.9, 0.1])      LB         ver
 
 ### #2, #5     

      weights = np.array([0.05, 0.05])    LB 0.865   ver27
      weights = np.array([0.1, 0.9])      LB 0.879   ver26
      weights = np.array([0.15, 0.85])    LB 0.878   ver28
      weights = np.array([0.2, 0.8])      LB 0.878   ver25
      weights = np.array([0.4, 0.6])      LB         ver
      weights = np.array([0.6, 0.4])      LB         ver
      weights = np.array([0.8, 0.2])      LB         ver
      weights = np.array([0.9, 0.1])      LB         ver
      
      
      
-------

## [Ensembling] [0.880] Audio Detection - 101
https://www.kaggle.com/mehrankazeminia/ensembling-0-880-audio-detection-101

### Data Set

      sub845 = pd.read_csv("../input/resnet34-more-augmentations-mixup-tta-inference/submission.csv")
      sub861 = pd.read_csv("../input/inference-tpu-rfcx-audio-detection-fast/submission.csv")
      sub877 = pd.read_csv("../input/resnet-wavenet-my-best-single-model-ensemble/submission.csv")


### Functions

      def generate(main, support, coeff):
          g1 = main.copy()
          g2 = main.copy()
          g3 = main.copy()
          g4 = main.copy()
    
          for i in main.columns[1:]:
              lm, Is = [], []                
              lm = main[i].tolist()
              ls = support[i].tolist() 
        
              res1, res2, res3, res4 = [], [], [], []          
              for j in range(len(main)):
                  res1.append(max(lm[j] , ls[j]))
                  res2.append(min(lm[j] , ls[j]))
                  res3.append((lm[j] + ls[j]) / 2)
                  res4.append((lm[j] * coeff) + (ls[j] * (1.- coeff)))
            
              g1[i] = res1
              g2[i] = res2
              g3[i] = res3
              g4[i] = res4
        
          return g1,g2,g3,g4


      def generate1(main, support, coeff):
    
          g = main.copy()    
          for i in main.columns[1:]:
        
              res = []
              lm, Is = [], []        
              lm = main[i].tolist()
              ls = support[i].tolist()  
        
              for j in range(len(main)):
                  res.append((lm[j] * coeff) + (ls[j] * (1.- coeff)))            
              g[i] = res
        
          return g

### Ensembling

      a1,a2,a3,a4 = generate(sub861, sub845, 0.80)
      b1,b2,b3,b4 = generate(sub877, a2, 0.85) 

      sub = b4
      sub.to_csv("submission.csv", index=False)

      b1.to_csv("submission1.csv", index=False)
      b2.to_csv("submission2.csv", index=False)
      b3.to_csv("submission3.csv", index=False)
      b4.to_csv("submission4.csv", index=False)

### Submission

      submission.csv     LB 0.880   ver1
      submission1.csv    LB 0.877   ver1
      submission2.csv    LB 0.478   ver1
      submission3.csv    LB 0.878   ver1
      submission4.csv    LB 0.880   ver1
      
### def generate(main, support, coeff):     
      
a1,a2,a3,a4 = generate(sub861, sub845, 0.80)

      b1,b2,b3,b4 = generate(sub877, a2, 0.80)  LB 0.879   ver2
      b1,b2,b3,b4 = generate(sub877, a2, 0.85)  LB 0.880   ver1   --- default
      b1,b2,b3,b4 = generate(sub877, a2, 0.90)  LB 0.879   ver3

-------


