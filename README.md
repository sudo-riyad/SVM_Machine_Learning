# SVM_Machine_Learning

## It is a Python based machine learning project wtih SVM classifier.

### LICENCE

Copyright © Riyad Ul Islam, Tanvir Hassan, Mou Saha

### Experiment Overview

The experiment were conducted on 5 object which are from object 1 to 5.To classify if it is object 1 or not, target class-1 is assigned for Object 1 and class-0 for others. And so all dataset is divided into two classes, either it is in Class-0 or in Class-1. For training all the objects data and then for testing data a graphical user interface has been created which is illustrated in Figure.
 
![gui](https://user-images.githubusercontent.com/57096728/145985878-122dfd4b-bc88-446c-aab0-11a443d5911d.JPG)

In the testing by defining a row index, a single object can be classified or all the objects can also be detected by selecting the test file. 
Before any kind of pre-processing Class-0 had total of 1150 and Class-1 had total of 315 data in below Table. 

| State                     | Class 0       | Class 1   |
| -------------             |:-------------:| -----:    |   
| Original Data             | 1150          | 315       |
| After Oversampling Method | 1150          | 575       |

This is where imbalance data be can be a problem as it is difficult to convince  your method to predict with better accuracy. In order to improve the performance random oversampling technique is implemented using RandomOverSampler Class and training datasets are selected randomly with replacement.
Its main objective is to balance class spreading through the random repetition of minority target instances. Oversampling is used for pre-processing the data in a way so that the number of minor class data become at least 50%.

### Documentation

Check the documentation for further understanding the project work : [Project Documentation](https://github.com/sudo-riyad/SVM_Machine_Learning/blob/b6197bfd7cc2791def361cbc710557d1bf21e821/Task_6_Team_1_Saha_Mou_Islam_Riyad-Ul-_Hasan_Tanvir.pdf)