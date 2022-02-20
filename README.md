# Deep-feature-extraction-from-CNNs-followed-by-optimization-with-GA    

## Project Description
This is a python-based project classifying the breast tumor tissue into benign or malignant, based on deep analysis of histopathological image samples of the popular `BreakHis` through the application of three popular `pre-trained` and `fine-tuned` Convolution Neural Networks, namely `GoogLeNet`, `ResNet-18`, and `VGG-19`, one at a time, followed by the extraction of the deep features from the CNN's `pre-final` layer and optimization (removal of redundant features) using the Genetic Algorithm (`GA`) for improved accuracy. The images are finally classified using any one of the `SVM`, `KNN` or `MLP` classifiers.

## Dataset description
The Breast Cancer Histopathological Image Classification (BreakHis) is composed of 9,109 microscopic images of breast tumor tissue collected from 82 patients using different magnifying factors (40X, 100X, 200X, and 400X). To date, it contains 2,480 benign and 5,429 malignant samples (700X460 pixels, 3-channel RGB, 8-bit depth in each channel, PNG format). This database has been built in collaboration with the P&D Laboratory – Pathological Anatomy and Cytopathology, Parana, Brazil (http://www.prevencaoediagnose.com.br). The dataset is available at:    
https://www.kaggle.com/ambarish/breakhis

## Classes of Division
In this project, we have used the histopathological image samples of human breast tissue, which have been classified into two categories, namely:  
- `Benign tissue`  
- `Malignant tissue` 

## Convolution Neural Network models used
Three CNN models have been applied on the dataset, namely:  
-	`GoogLeNet`  
-	`ResNet-18`  
-	`Visual Geometry Group (VGG-19)`

## Classifier models used inside the Genetic Algorithm (GA)
Three classifier models have been used, namely:  
-	`Support Vector Machines (SVM) (RBF Kernel)`  
-	`K-Nearest Neighbors (KNN) (K=2 used)`  
-	`Multi-Layer Perceptron (MLP)`  

## 'Accuracy' vs 'Generation' plots
The following plots show the variation in accuracy of our validation dataset with the increasing number of generations after the application of Genetic Algorithm (GA) in each of the three Convolutional Neural Networks. The 'KNN' classifier has been used in every case.     
No. of Epochs: `5`, No. of Generations: `10`
-     GoogleNet using KNN classifier
     ![image](https://user-images.githubusercontent.com/84792746/154838869-35486f58-74c8-46b3-9b15-75333ddd2eef.png)
-     ResNet-18 using KNN classifier
     ![image](https://user-images.githubusercontent.com/84792746/154838818-d415193f-1736-4272-a9c3-90ce013bfc37.png)
-     VGG-19 using KNN classifier
     ![image](https://user-images.githubusercontent.com/84792746/154838834-71b65cea-cc2b-4527-b6ad-0d01e2c532c5.png)
     
## Flow diagram of the Genetic Algorithm


![image](https://user-images.githubusercontent.com/84792746/154852688-200dd978-ec4a-47c1-b073-0d3249e3d89c.png)


## Dependencies
Since the entire project is based on `Python` programming language, it is necessary to have Python installed in the system. It is recommended to use Python with version `>=3.6`.
The Python packages which are in use in this project are  `matplotlib`, `numpy`, `pandas`, `scikit-learn`, `torch` and `torchvision`. All these dependencies can be installed just by the following command line argument
- pip install `requirements.txt`

## Code implementation
- ### Data paths :
      Current directory -------> data
                                  |
                                  |
                                  |               
                                  --------------------->  train
                                  |                         |
                                  |             -------------------------
                                  |             |        |              |
                                  |             V        V              V
                                  |           class_1  class_2 ...... class_n
                                  |
                                  |
                                  |              
                                  --------------------->   val
                                                            |
                                                -------------------------
                                                |        |              |
                                                V        V              V
                                              class_1  class_2 ...... class_n
                                              
                               
- Where the folders `train` and `val` contain the folders `benign` and `malignant`, which include the original histopathological images of respective type of human breast tumor tissue in `.jpg`/`.png` format.

- ### Training and Evaluation :

          usage: main.py [-h] [-data DATA_FOLDER] [-classes NUM_CLASSES]
                         [-ext EXTRACTOR_TYPE] [-classif CLASSIFIER_TYPE]

          Application of Genetic Algorithm

          optional arguments:
            -h, --help            show this help message and exit
            -data DATA_FOLDER, --data_folder DATA_FOLDER
                                  Path to data
            -classes NUM_CLASSES, --num_classes NUM_CLASSES
                                  Number of data classes
            -ext EXTRACTOR_TYPE, --extractor_type EXTRACTOR_TYPE
                                  Choice of deep feature extractor
            -classif CLASSIFIER_TYPE, --classifier_type CLASSIFIER_TYPE
                                  Choice of classifier for GA
        
-  ### Run the following for training and validation :
  
      `python main.py -data data -classes n -ext resnet -classif KNN`
      
-  ### Specific tokens :

          GoogLeNet: 'googlenet'
          ResNet-18: 'resnet'
          VGG-19: 'vgg16'
          SVM Classifier: 'SVM'
          KNN Classifier: 'KNN'
          MLP Classifier: 'MLP'
