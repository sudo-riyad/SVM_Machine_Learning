from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import csv
import os
from tkinter import ttk
import time
import tkinter as tk
import tkinter.scrolledtext as tkst
################
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import os
from glob import glob
import pickle

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

import librosa

############## Intalling Library #########

# pip install -U scikit-learn
# pip install pandas
# pip install matplotlib
# pip install imbalanced-learn
# pip install librosa
# pip install openpyxl

class SVM_Classifier:
    model = None

    def __init__(self,filePath,startIndex,endIndex,testOutput,trainOutput):
        self.filePath = filePath
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.alldataset = pd.DataFrame()
        self.testOutput = testOutput
        self.trainOutput = trainOutput

###### Read all data from data directory and start the training process #######
    def startTraining(self):
        self.read_all_data(self.filePath,self.startIndex,self.endIndex)
        self.re_assign_targetClass()
        #view_signal()
        print("in main")
        self.training()   

###### Returns Target Class #######
    def getTargetClass(self,fileName):
        object_type = 0
        if 'Object 1' in fileName:
            object_type = 1
        elif 'Object 2' in fileName:
            object_type = 2
        elif 'Object 3' in fileName:
            object_type = 3
        elif 'Object 4' in fileName:
            object_type = 4
        elif 'Object 5' in fileName:
            object_type = 5
        else :
            object_type = 6

        return object_type

###### Read all data from data directory and loading those data in data frame #########
    def read_all_data(self,data_dir,startIndex,endIndex):

        folders = glob(data_dir+"/*")
        print(folders)
        object_type = 0
        final_data =pd.DataFrame()

        for f in range(0,len(folders)):
        
            
            files = glob(folders[f]+"/*")
            
            fileName = folders[f]
        
            object_type = self.getTargetClass(fileName)
            for file in range(0,len(files)):
                fileName = files[file]    
                self.preprocessing(fileName,object_type,startIndex,endIndex)
                

###### Map target class 1 with object 1 and target class 0 with not object 1 #######
    def re_assign_targetClass(self):
        
        mask = self.y_values(self.alldataset)
        self.alldataset.loc[ mask!=1, 0] = 0

        print("After reassigning the target class, number of samples of each class is...")
        print(Counter(self.alldataset.loc[mask != 1 , 0]))
        print(Counter(self.alldataset.loc[mask == 1 , 0]))
        print("class 1 -> Object 1 AND class 0 -> not Object 1")

###### Over sampling to reduce imbalanceness in the data set ########
    def over_sampling(self,x,y):
        
        oversampler = RandomOverSampler(sampling_strategy=0.5)
        x_over,y_over = oversampler.fit_resample(x,y)
        return x_over, y_over
###### Preprocessing Data ######        
####### All the function call for feature extraction, over sampling , feature scaling and spliting data set
    def get_train_test_data(self):
       
        print(self.alldataset.shape)
        newXValue = self.featureExtraction(self.x_values(self.alldataset))
        print("feature extraction is done")
        y_val = self.y_values(self.alldataset)
        x_over , y_over = self.over_sampling(newXValue,y_val)
        print("over sampling is done")
        x_over = self.featureScaling(x_over)
        print("feature scaling is done")
        X_train, X_test, y_train, y_test = self.split_train_test(x_over,y_over)
        print("train test split is done")
        return X_train, X_test, y_train, y_test

##### 70% data for training and 30 % for testing #######
    def split_train_test(self,x,y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=0)
        return X_train, X_test, y_train, y_test

###### concatinating  to central data set ########
    def append_to_dataset(self,dataset):
        self.alldataset = pd.concat([self.alldataset, dataset],ignore_index=True)
###### Data preprocessing ########
    def preprocessing(self,filename,target,startingIndex=7, endingIndex=3406):
        
        rawData = pd.read_excel(filename,header=None)
        rawData = rawData.iloc[:,startingIndex:(endingIndex+1)]

        result = pd.Series([target for x in range(0,len(rawData.index)) ]) #adding result class
        
        rawData.insert(0,0,result) #adding result class at 0th column
        rawData.columns = [i for i in range(0,len(rawData.columns))]
        
        self.append_to_dataset(rawData)
        

    def y_values(self,dataset):
        return dataset.iloc[:,0]

    def x_values(self,dataset):
        return dataset.iloc[:,1:]

###### extract feature for single signal #########       
    def extract_single_feature(self,signal_t):
        stft = np.abs(librosa.stft(signal_t))
        mfccs = np.mean(librosa.feature.mfcc(y=signal_t, sr=22050, n_mfcc=40).T,axis=0)
        return mfccs

####### Feature extraction for whole data ########
    def featureExtraction(self,dependent_var):
        print("feature extraction starts here ....")
        newdata = pd.DataFrame(columns=[i for i in  range(0,40)])
        
        for row in range(0,len(dependent_var)) :
            item = dependent_var.iloc[row]
            item = item.values
            mfcc = self.extract_single_feature(item)
            mfccdp = pd.Series(mfcc)
            newdata = newdata.append(mfccdp,ignore_index=True )
        return newdata

###### Scaling features #######
    def featureScaling(self,dependent_var):
        print("feature scaling starts here ...")
        sc = StandardScaler()
        dependent_var = sc.fit_transform(dependent_var)
        dependent_var = pd.DataFrame(dependent_var)
        return dependent_var

    def singleFeatureScaling(self,item):
        sc = StandardScaler()
        dependent_var = sc.transform(item)
        dependent_var = pd.DataFrame(dependent_var)
        return dependent_var

###### Plotting signal and other processing phase ######
    def view_signal(self,signal):
        
        fig1 = plt.figure("Feature Extraction")
        
        sp1 = fig1.add_subplot(2,2,1)
        sp2 = fig1.add_subplot(2,2,2)
        sp3 = fig1.add_subplot(2,2,3)
    
        sp1.title.set_text("Original signal")
        sp2.title.set_text("MFCC  ")
        sp3.title.set_text("Scaled MFCC")
        sp1.set_xlabel('time(s)')
        sp1.set_ylabel('amplitute')
        sp1.plot(signal)
        mfcc = self.extract_single_feature(signal.values)
        mfcc = mfcc.reshape(-1,1)
        sp2.plot(mfcc)
        scaled_mfcc = self.featureScaling(mfcc)
        sp3.plot(scaled_mfcc)
        plt.show()

####### saving trained model ######
    def saveModel(self,_model):
        with open('svm_classifier.pkl','wb') as file:
            print("model saved svm_classifier.pkl")
            return  pickle.dump(_model,file)   

####### loading trained model ######    
    def loadModel(self):
        with open('svm_classifier.pkl','rb') as file:
            return  pickle.load(file)  

####### preparing data for testing ######  
    def prepareForTesting(self,filename,startIndex,endIndex):
        print("preparing for testing")
        print(filename,startIndex,endIndex)
        dataset = pd.read_excel(filename,header=None)
        if(startIndex == endIndex ) and startIndex == 0:
            endIndex = len(dataset.columns)
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.alldataset = dataset.iloc[:,startIndex:endIndex]

        newXValue = self.featureExtraction(self.alldataset)
        print("feature extraction is done")
        scaled_data = self.featureScaling(newXValue)
        print("feature scaling is done")
        self.scaledDataset = scaled_data

        global model
        model = self.loadModel()
        print("ready for single testing") 

####### Testing any data and predict the source ########
    def testing_all_new_data(self):
        print("testing all will start")
        global model
        model = self.loadModel()
        y_pred = model.predict(self.scaledDataset)
        output = ''
        for i in range(0,len(y_pred)):
            if y_pred[i] ==1:
                output = output + '{} {} - Object 1\n'.format(y_pred[i],i)
            else:
                output = output + '{} {} - Not Object 1\n'.format(y_pred[i],i)
        self.testOutput.delete(1.0,END)
        self.testOutput.insert(1.0,output)

    def singleTesting(self, index):
        global model
        print(self.scaledDataset.shape)
        
        data = self.scaledDataset.iloc[index,:]
        pred = model.predict([data])
        output = ''
        if pred[0] ==1:
            output = '{} Object 1\n'.format(index)
        else:
            output = '{} Not Object 1\n'.format(index)
        print(pred)
        self.testOutput.insert(END,output)
        self.view_signal(self.alldataset.iloc[index,:])
        
###### training and validation ########       
    def training(self):

        print("train test will be starting")
        X_train, X_test, y_train, y_test = self.get_train_test_data()
        print("training starts....")
        #training

        clf = svm.SVC(kernel='rbf') 
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        self.saveModel(clf)
        self.modelAnalysis(y_test,y_pred)

        
###### plotting ROC curve ######
    def plot_roc_curve(self, fpr, tpr):
        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

###### Analysis and evaluating the classifier ########
    def modelAnalysis(self, y_test,y_pred):

        cnf_matrix = metrics.confusion_matrix(y_test, y_pred) 
        print("confusion matrix")
        print(cnf_matrix)
        #plt.imshow(cnf_matrix, cmap='binary')

        report = metrics.classification_report(y_test,y_pred)
        print("svm classifier report")
        print(report)

        TN,FP,FN,TP = metrics.confusion_matrix(y_test, y_pred).ravel()
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)
        # Overall accuracy for each class
        ACC = (TP+TN)/(TP+FP+FN+TN)
        f_score = 2*(PPV*TPR)/(PPV+TPR)
        output = "TP {}\nTN {}\nFP {}\nFN {}\nFDR {:.4f}\nNPV {:.4f}\nTPR {:.4f}\nTNR {:.4f}\nACC {:.2f}%\nf1 {:.2f}".format(TP,TN,FP,FN,FDR,NPV,TPR,TNR,ACC*100,f_score)
        self.trainOutput.delete(1.0,END)
        self.trainOutput.insert(1.0, output)
        fpr,tpr,threshold =  metrics.roc_curve(y_test, y_pred) 
        self.plot_roc_curve(fpr,tpr)

##### placeholder ######
classifire = SVM_Classifier("",1,5,"","") 

##### GUI event processing ######
def makeView():

    

    def dir_selected(file_path):
        data_directory = askdirectory()
        print(data_directory)
        file_path.delete(0,END)
        file_path.insert(0,data_directory)

    def stringToInt(data):
        try:
            intdata = int(data)
            return intdata
        except:
            return 0

    def training_btn_clicked(file_path,start_entry,end_entry,resultArea):
        print("Training starts here ....")
        path = file_path.get()
        start_index = stringToInt(start_entry.get())
        end_index = start_index + stringToInt(end_entry.get())
        print(path,start_index,end_index)
        svm = SVM_Classifier(path,start_index,end_index,"",resultArea)
        svm.startTraining()

    def testing_btn_clicked(file_path,start_entry,end_entry,row_entry):
        row = stringToInt(row_entry.get())
        global classifire
        classifire.singleTesting(row)

    def testing_all_btn_clicked(test_file_path,start_test_entry,end_test_entry):
        classifire.testing_all_new_data()

    def file_selected(file_path):
        fileName = askopenfilename()
        print(fileName)
        file_path.delete(0,END)
        file_path.insert(0,fileName)

    def prepareModel(file_path,start_entry,end_entry,testResult):
        path = file_path.get()
        start_index = stringToInt(start_entry.get())
        end_index = start_index + stringToInt(end_entry.get())
        print(path,start_index,end_index)
        global classifire
        classifire = SVM_Classifier(path,start_index,end_index,testResult,"")
        classifire.prepareForTesting(path,start_index,end_index)

######### GUI creating ########

    gui = Tk()
    gui.title('Discrimination of reflected sound signals')
    learing_frame = Frame(gui, bg='bisque2',  padx=3,pady=3)
    testing_frame = Frame(gui, bg='lavender',  padx=3, pady=3)
    learing_frame.grid(row=0, column=0, sticky="ns")
    testing_frame.grid(row=0, column=1, sticky="nsew")
    open_btn=Button(learing_frame, text='Select Data directory',cursor="hand2",bg='red',command=lambda: dir_selected(file_path))
    open_btn.grid(row=0, column=0)
    file_path = Entry(learing_frame,text='',width = 50)
    file_path.grid(row=0, column=1, columnspan = 2)

    train_btn = Button(learing_frame, text='Train',cursor="hand2",bg='OliveDrab1',width = 15,command= lambda: training_btn_clicked(file_path,start_entry,end_entry,resultArea))
    train_btn.grid(row=20, column=0)

    starting=Label(learing_frame, text='Starting Index', bg='bisque2')
    starting.grid(row=10, column=0)
    start_entry = Entry(learing_frame,text='')
    start_entry.grid(row=10, column=1)

    ending=Label(learing_frame, text='Signal Length', bg='bisque2')
    ending.grid(row=15, column=0)
    end_entry = Entry(learing_frame,text='')
    end_entry.grid(row=15, column=1)

    resultArea = tkst.ScrolledText(learing_frame,height=15,width=20)
    resultArea.grid(padx=10, pady=10,row=60,column = 1, sticky=tk.W)

    name_learning=Label(learing_frame, text='Training Data',font=("times new roman",13,"bold"),bg='maroon1')
    name_learning.grid(row=85, columnspan=2)
    name_testing=Label(testing_frame, text='Testing Data',font=("times new roman",13,"bold"),bg='purple1')
    name_testing.grid(row=70, columnspan=3)
    testResultArea = tkst.ScrolledText(testing_frame,height=15,width=20)
    testResultArea.grid(padx=10, pady=10,row=50, column = 1, sticky=tk.W)

    right = 0
    width = 1
    open_file=Button(testing_frame, text='Select Data File',cursor="hand2",bg='red',command=lambda: file_selected(test_file_path))
    open_file.grid(row=0, column=right)
    test_file_path = Entry(testing_frame,text='',width = 50)
    test_file_path.grid(row=0, column=1, columnspan = 2)

    test_btn = Button(testing_frame, text='Test',cursor="hand2",bg='MistyRose1',width = 15,command= lambda: testing_btn_clicked(test_file_path,start_test_entry,end_test_entry,row_entry))
    test_btn.grid(row=25, column=1)
    test_all_btn = Button(testing_frame, text='Test all ',cursor="hand2",bg='SeaGreen1',width = 15,command= lambda: testing_all_btn_clicked(test_file_path,start_test_entry,end_test_entry))
    test_all_btn.grid(row=25, column=2)

    starting_test_index=Label(testing_frame, text='Starting Index', bg='lavender')
    starting_test_index.grid(row=10, column=0)
    start_test_entry = Entry(testing_frame,text='')
    start_test_entry.grid(row=10, column=1)

    ending_test=Label(testing_frame, text='Signal Length', bg='lavender')
    ending_test.grid(row=15, column=right)
    end_test_entry = Entry(testing_frame,text='')
    end_test_entry.grid(row=15, column=1)
    row = Label(testing_frame, text='Row Index', bg='lavender')
    row.grid(row=20, column=0)

    prepareBtn = Button(testing_frame,text="Prepare model",cursor="hand2",bg='OliveDrab1',command= lambda: prepareModel(test_file_path,start_test_entry,end_test_entry,testResultArea))
    prepareBtn.grid(row = 25,column=0)

    row_entry = Entry(testing_frame,text='')
    row_entry.grid(row=20, column=1)


    gui.mainloop()


if __name__ == "__main__":
    makeView()
