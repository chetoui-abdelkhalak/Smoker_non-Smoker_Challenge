# Smoker_non-Smoker_Challenge

# Ramp Starting Kit on Body signals for smoking Classification

Authors: EZZAHERY Ayoub, CHETOUI Abdelkhalak, BOUZINAB Badr

There are several body signals that tell if a person is a smoker or not, either short-term ones or long-term ones. Here in this project, we are going to work on a classification task to determine if the person is a smoker or not given their body signals. 

***PS: the dataset was taken from kaggle***

This RAMP work will be divided into three parts: A first step will be discovering the dataset to get insight on the differents patterns. Next, we will preprocess the data since we have two types of columns (numerical and categorial). A final step will be testing different machine learning algorithms based on different metrics.


### 1-Set up:

This ramp project requires the following dependencies:


1-Downloading the ramp workflow (if not already done)
```
$ pip install ramp-workflow
```
 
 2-Downloading the requirements:
```
$ pip install -r requirements.txt
```

### 2-Get Started:

*The dataset is available in the data folder.*

The starting kit consists on 2 classifiers:

-Logistic Regression

-KNN

-XGBoost

To test the starting kit, you can run:
```
$  ramp-test --submission starting_kit
```
