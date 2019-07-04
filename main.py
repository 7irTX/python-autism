import pandas as pd
import numpy as np
import config as cf

import sys
import numpy
numpy.set_printoptions(threshold = sys.maxsize)

from indx_features import indx_ID
from sp_quicksort import quickSort
from Diagnosis_class import Diagnosis_class
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from math import pow

from sklearn import svm #main
from sklearn.feature_selection import RFE #main
from sklearn.svm import SVC #main
from sklearn.utils.validation import column_or_1d #mian
from sklearn.model_selection import train_test_split #main
from sklearn.metrics import recall_score #main
from random import shuffle
from sklearn.model_selection import GridSearchCV

from sklearn import datasets #test
from sklearn.decomposition import PCA #test
from sklearn.datasets import make_classification #test
from sklearn.feature_selection import RFECV #test
from sklearn.model_selection import StratifiedKFold #test
iris = datasets.load_iris()

from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
#from sklearn.pipeline import Pipeline
my_list = []
speaker_list = []

def prediction_svm(X_train,y_train,X_predict,y_actual,ker,ker_coee,C):
        #ker = 'linear'
        #clf = svm.SVC(gamma='auto',kernel = ker)
        clf = svm.SVC(gamma = ker_coee, kernel = ker, C = C,verbose = False)
        #sklearn.model_selection.GridSearchCV
        clf.fit(X_train, np.ravel(y_train,order='C'))
        pred = clf.predict(X_predict)
        #N = np.shape(pred)[0]
        '''
        TP = 0 
        FP = 0 
        FN = 0 
        TN = 0
        '''
        #UAR evaluation
        # print(pred)
        #UAR metrics.balanced_accuracy_score(y_true, y_pred)
        '''
        for i in range(N):
                if pred[i] == 1 and y_actual[i] == 1:
                        TP = TP + 1
                if pred[i] == 1 and y_actual[i] == 2:
                        FP = FP + 1
                if pred[i] == 2 and y_actual[i] == 1:
                        FN = FN + 1
                if pred[i] == 2 and y_actual[i] == 2:
                        TN = TN + 1
        print(TP,FP,TN,FN)
        if TP + FP == 0:
                R1 = -1
                return TN/(TN+FN)
        else:
                R1 = TP/(TP+FP)
                
        if TN + FN == 0:
                R2 = -1
                return R1
        else:
                R2 = TN/(TN+FN)
                
        
        UAR = (R1+R2)/2
        '''
        UAR = recall_score(y_actual,pred,average = 'macro')
        #f = open("result.txt","a")
        #f.write("\n" + ker + " " + ker_coee + " " + str(C) + "\n" + np.array2string(y_actual.T) + "\n\n" + np.array2string(pred.T))
        #f.close()
        return UAR
        
        
def dim_elimination(Features, labels):
        #RFE = Ranking recursive feature elimination
        ker = "linear"
        C_value = 1
        svc = SVC(kernel = ker, C = C_value)
        #rfe = RFE(estimator = svc, n_features_to_select = N_features, step = N_steps)
        rfe = RFECV(estimator = svc, step = 1, cv = StratifiedKFold(n_splits = 5, shuffle = False), verbose = False)
        #N_labels = np.shape(task)[1]
        #rfe = RFECV(estimator=svc, step=N_steps, cv=StratifiedKFold(2),scoring='accuracy')
        labels = binary_array(labels)
        rfe.fit(Features, labels)
        ranking = rfe.ranking_
        return ranking

def ranking(path1,path2,task):
        label = read(path1)
        inp = read(path2)
        y = np.array(label)
        X_tmp = np.array(inp)
        size = np.shape(X_tmp)[1]
        X = X_tmp[:,2:size]
        rank = dim_elimination(X,np.ravel(y,order = 'C'))
        '''
        if task == "binary":
                rank = dim_elimination(X,np.ravel(y,order='C'),"binary")
        if task == "multiple":
                rank = dim_elimination(X,np.ravel(y,order='C'),"multiple")
        '''
        return rank

def getFeatures(X):
        size = np.shape(X)[1]
        return X[:,2:size]

def binary_array(array):
        N = np.shape(array)[0]
        var = np.zeros(N)
        for i in range(N):
                if array[i] > 1:
                        var[i] = 2
                else:
                        var[i] = 1
        return var

def group_features(train_class, test_class):
        train_features = []
        train_labels = []
        test_features = []
        test_labels = []
        train_genders = []
        test_genders = []

        train_size = np.shape(train_class)[0]
        test_size = np.shape(test_class)[0]


        for i in range(train_size):
                feature_size = np.shape(train_class[i].features)[0]
                for j in range(feature_size):
                        #train_features = np.vstack((train_class[i].features,train_features))
                        train_features.append(train_class[i].features[j])
                        train_labels.append(train_class[i].label[j])
                train_genders.append(train_class[i].gender)
        
        for i in range(test_size):
                test_feature_size = np.shape(test_class[i].features)[0]
                for j in range(test_feature_size):
                        test_features.append(test_class[i].features[j])
                        test_labels.append(test_class[i].label[j])
                test_genders.append(test_class[i].gender)
        return np.array(train_features), np.array(train_labels), np.array(test_features), np.array(test_labels) 

def k_fold_division(my_list, indexes):
        k_folds = 6

        tmp_length = np.shape(indexes)[0]
        for i in range(tmp_length):
                indexes[i] = indexes[i] - 1

        N_size = 11
        block = []
        for i in range(k_folds):
                tmp = []
                for j in range(N_size):
                        tmp.append(my_list[indexes[(i)*N_size + j]])
                block.append(tmp)

        return block

def merge_classes(matrix, indexes, N_times):
        #assume all columns/rows have same size
        size = np.shape(matrix[0])[0]
        out = []
        for i in range(N_times):
                tmp_index = indexes[i]
                for j in range(size):
                        out.append(matrix[tmp_index][j])
        
        return out

def exhaustive_search(my_list):
        kernel_coee = ['scale','auto','auto_deprecated']
        kernel = ["linear","rbf","sigmoid"]
        uars = []
        #not stable      #indexes = [53, 65, 4, 62, 61, 63, 25, 37, 64, 46, 23, 26, 13, 55,39,57,5,3,41,11,38,12,34,16,31,56,29,51,60,8,28,18,33,54,42,14,19,40,36,6,58,49,32,7,27,9,59,21,43,67,2,45,17,30,15,35,10,24,52,47,44,50,22,1,66,48,20]
        #not stable      #indexes = [27,28,10,34,56,46,25,22,62,58,19,5,40,61,15,43,11,9,59,2,52,64,45,39,44,12,31,65,51,41,37,18,21,57,32,47,63,8,38,4,13,24,7,30,53,36,50,35,16,33,66,20,6,48,67,23,54,60,17,1,49,26,42,55,3,14,29]
        
        #indexes = [50,53,43,3,9,26,34,29,32,31,2,40,8,19,33,20,42,28,37,57,38,30,55,48,51,7,16,21,66,65,1,35,54,61,12,41,22,36,5,47,58,14,11,17,62,15,59,63,64,4,45,46,18,60,49,10,52,56,13,6,67,23,24,39,27,25,44]
        #indexes = [44,38,29,51,54,36,22,16,52,14,17,26,28,39,64,25,27,20,40,19,31,12,63,34,8,56,48,58,60,1,65,10,45,30,15,37,11,55,59,46,18,35,13,41,53,42,2,66,24,50,4,62,7,23,33,43,61,32,5,9,57,49,67,3,47,21,6]
        
        #best indexes
        indexes = [54,38,10,40,1,46,67,39,29,13,2,16,11,43,65,41,25,21,59,24,49,3,37,62,15,64,9,20,6,55,31,30,34,8,53,44,63,4,19,60,14,26,36,58,17,23,48,35,12,47,42,22,5,18,45,27,52,28,33,7,56,66,32,57,51,50,61]
        #combinations = [[1,2,3,4,5],[0,2,3,4,5],[0,1,3,4,5],[0,1,2,4,5],[0,1,2,3,5],[0,1,2,3,4]]
        #combinations = [[0,1,2,3],[0,1,2,4],[0,1,3,4],[1,2,3,4],[0,2,3,4]] #fold 5 leaves for testing
        combinations = [[0,1,2,3,4]]
        test_folds = 5
        blocks = k_fold_division(my_list, indexes)

        #test_set = np.array(test_set)
        #train_set = np.array(train_set)
        N_coee = 1
        N_kernel = 1
        N_C = 1
        #C = [1,2,3,4,5] #how much you want to avoid misclassifying each training example
        times = 1


        #N_test = np.shape(test1)[0]
        #N_train = np.shape(train1)[0]
        #N_merge = 6
        
        '''
        for i in range(N_merge):
                tmp = []
                for j in range(N_test):
                        #print(test_set[i][j])
                        tmp.append(my_list[test_set[i][j]])
                        #np.vstack(my_list[test_set])
                test_class.append(tmp)
                tmp = []
                for j in range(N_train):
                        tmp.append(my_list[train_set[i][j]])
                train_class.append(tmp)
        '''
        #print(np.shape(test_class), np.shape(train_class))
        #return



        for i in range(N_coee):
                for j in range(N_kernel):
                        C = 1
                        for k in range(N_C):
                                '''
                                if C < 1:
                                        C = 1e-10 * pow(10,k)
                                        C = C * 10
                                else:
                                        C = C + 1
                                C = 2
                                '''
                                #print(C)
                                for n in range(times): #refold my_list
                                        test_class = blocks[5] #last fold for testing, others for training
                                        train_class = []
                                        #train_class = merge_classes(blocks, combinations[n],times - 1)
                                        train_class = merge_classes(blocks, combinations[n],test_folds)
                                        train_features, train_labels, test_features, test_labels = group_features(train_class,test_class)
                                        train_labels = binary_array(train_labels)
                                        test_labels = binary_array(test_labels)
                                        tmp_uar = prediction_svm(train_features,train_labels,test_features,test_labels,kernel[j],kernel_coee[i],C)
                                        uars.append(tmp_uar)
                                        #print(np.shape(train_class)[0], np.shape(test_class)[0])

                                        #train class, test class
                                        #train_class,test_class = train_test_split(my_list, test_size = 0.15)
                                        
                                        
                                        #split class->training by prediction svm->find appropriate coefficient exhaustively->dimension elimination
        
        return uars
        

#13/06 grouping features and labels to an array
def class_init():
        tmp_list = []
        '''
        path_label = "C:/Users/Yitian Chen.DESKTOP-TIIP443/Desktop/tmp1/training/result_csv/all_label.csv"
        path_result = "C:/Users/Yitian Chen.DESKTOP-TIIP443/Desktop/tmp1/training/result_csv/all_result.csv"
        path_speakerID = "C:/Users/Yitian Chen.DESKTOP-TIIP443/Desktop/UNSW_Thesis/ComParE2013_Autism/lab/ComParE2013_Autism.csv"
        path_gender = "C:/Users/Yitian Chen.DESKTOP-TIIP443/Desktop/UNSW_Thesis/ComParE2013_Autism/lab/gender_lable.csv"
        '''
        path_label = "./result_csv/all_label.csv"
        path_result = "./result_csv/all_result.csv"
        path_speakerID = "./lab/ComParE2013_Autism.csv"
        path_gender = "./lab/gender_lable.csv"
        tmp_label = read(path_label)
        tmp_result = read(path_result)
        tmp_speaker = read(path_speakerID)
        tmp_gender = read(path_gender)

        label = np.array(tmp_label)
        features = np.array(tmp_result)
        speakerID = np.array(tmp_speaker)[:,1]
        gender = np.array(tmp_gender)
        N1 = np.shape(label)[0]
        N2 = cf.NUM_SPEAKERS
        #print(N2)
        for i in range(N1):
                tmp_list.append(Diagnosis_class(label[i],speakerID[i],features[i],gender[i]))
        quickSort(tmp_list,0,N1-1)
        my_list.append(tmp_list[0])
        j = 0
        for i in range(1,N1):
                if(tmp_list[i-1].speaker_ID) == (tmp_list[i].speaker_ID):
                        my_list[j].label = np.vstack((tmp_list[i].label,my_list[j].label))
                        my_list[j].features = np.vstack((tmp_list[i].features,my_list[j].features))
                        my_list[j].gender = np.vstack((tmp_list[i].gender,my_list[j].gender))
                else:
                        my_list.append(tmp_list[i])
                        j = j + 1

        #x,y = train_test_split(my_list,test_size = 0.15, random_state = 10)
        #x,y = train_test_split(my_list,test_size = 0.15)
        #train_features, train_labels, test_features, test_labels = group_features(x,y)
        return my_list
        #return train_features, train_labels, test_features, test_labels
        #print(np.shape(y))

        #random state = initialization of random seed
        #dim x[t].features is (a,88), which a indicates the number of speakers and 88 is the num of features
        
        #may not need group label together

def select_best_features(features):
        #index = [1, 10, 12, 14, 15, 23, 30, 31, 39, 41, 49, 53, 59, 61, 62, 66, 69, 78, 80, 81, 82, 83]
        index = [1, 2, 10, 11, 12, 13, 14, 15, 32, 33, 34, 39, 41, 43, 45, 47, 49, 53, 59, 61, 62, 64, 66, 78, 81, 82, 83, 84, 85, 87]
        #index = [1, 10, 12, 14, 15, 39, 41, 49, 53, 59, 61, 62, 66, 78, 81, 82, 83]
        out = []
        N = np.shape(index)[0]
        for i in range(N):
                out.append(features[index[i]])
        
        return out


def reshape_class_init():
        tmp_list = []
        '''
        path_label = "C:/Users/Yitian Chen.DESKTOP-TIIP443/Desktop/tmp1/training/result_csv/all_label.csv"
        path_result = "C:/Users/Yitian Chen.DESKTOP-TIIP443/Desktop/tmp1/training/result_csv/all_result.csv"
        path_speakerID = "C:/Users/Yitian Chen.DESKTOP-TIIP443/Desktop/UNSW_Thesis/ComParE2013_Autism/lab/ComParE2013_Autism.csv"
        path_gender = "C:/Users/Yitian Chen.DESKTOP-TIIP443/Desktop/UNSW_Thesis/ComParE2013_Autism/lab/gender_lable.csv"
        '''
        path_label = "./result_csv/all_label.csv"
        path_result = "./result_csv/all_result.csv"
        path_speakerID = "./lab/ComParE2013_Autism.csv"
        path_gender = "./lab/gender_lable.csv"
        tmp_label = read(path_label)
        tmp_result = read(path_result)
        tmp_speaker = read(path_speakerID)
        tmp_gender = read(path_gender)

        label = np.array(tmp_label)
        features = np.array(tmp_result)
        speakerID = np.array(tmp_speaker)[:,1]
        gender = np.array(tmp_gender)
        N1 = np.shape(label)[0] #N1 = 67
        N2 = cf.NUM_SPEAKERS
        N_min = 15
        #print(N2)

        for i in range(N1):
                tmp_fea = select_best_features(features[i])
                #tmp_fea = features[i]
                tmp_list.append(Diagnosis_class(label[i],speakerID[i],tmp_fea,gender[i]))
                #tmp_list.append(Diagnosis_class(label[i],speakerID[i],features[i],gender[i]))
        quickSort(tmp_list,0,N1-1)
                
        #tmp_list[0].features = select_best_features(tmp_list[0].features)
        my_list.append(tmp_list[0])
        j = 0

        count = 0
        for i in range(1,N1):
                if(tmp_list[i-1].speaker_ID) == (tmp_list[i].speaker_ID):
                        #if count < N_min:
                        my_list[j].label = np.vstack((tmp_list[i].label,my_list[j].label))
                        best_tmp = tmp_list[i].features
                        my_list[j].features = np.vstack((best_tmp,my_list[j].features))
                        my_list[j].gender = np.vstack((tmp_list[i].gender,my_list[j].gender))
                        count = count + 1
                else:
                        my_list.append(tmp_list[i])
                        j = j + 1
                        #count = 0

        return my_list


def cross_validation(train_features, train_labels, test_features, test_labels):
        parameters = {'kernel':('linear','rbf','sigmoid'), 'C' : [1,10]}
        svc = svm.SVC(gamma = "scale")
        out = GridSearchCV(svc,parameters, cv = 5)
        out.fit(train_features,train_labels)
        print(sorted(out.cv_results_.keys()))


def main():
        path1 = "C:/Users/Yitian Chen.DESKTOP-TIIP443/Desktop/tmp1/training/result_csv/devel_lable.csv"
        path2 = "C:/Users/Yitian Chen.DESKTOP-TIIP443/Desktop/tmp1/training/result_csv/result_devel.csv"
        path3 = "C:/Users/Yitian Chen.DESKTOP-TIIP443/Desktop/tmp1/training/result_csv/result_train.csv"
        path4 = "C:/Users/Yitian Chen.DESKTOP-TIIP443/Desktop/tmp1/training/result_csv/train_lable.csv"
        label_devel = read(path1)
        inp_devel = read(path2)
        y_train = np.array(label_devel)
        X_train = np.array(inp_devel)
        X_train = getFeatures(X_train)
        
        label_train = read(path4)
        inp_train = read(path3)
        y_actual = np.array(label_train)
        X_predict = np.array(inp_train)
        X_predict = getFeatures(X_predict)
        #UAR = prediction_svm(X_train,binary_array(y_train),X_predict,binary_array(y_actual))
        #print(UAR)

        rank = ranking(path1,path2,"binary")
        #print(rank)
        
def main2():
        my_list = class_init()
        #my_list = reshape_class_init()
        #print(np.shape(my_list[1].features))
        uar = exhaustive_search(my_list)
        print(uar)


def my_split(my_list,chosen):
        N = np.shape(my_list)[0]
        start = 0
        end = N
        if start == chosen:
                start = 2
        if end == chosen:
                end = N-1
        y1 = my_list[start:chosen]
        y2 = my_list[chosen+1:end]
        x = my_list[chosen]
        return y1,y2,x

def test_split():
        X = [1,2,3,4,5,6]
        for i in range(6):
                y1,y2,x = my_split(X,i)
                print(y1,y2,x)

def test_array():
        x =     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,]

        y =     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1,
        2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1,
        2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 2,
        1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2,]
        uar = recall_score(np.array(x),np.array(y),average = 'macro')
        print(uar)

def k_fold_cross_validation(items, par, randomize=False):
        size = np.shape(items)[0]
        if randomize:
                items = list(items)
                shuffle(items)
        mid = int(size * par)
        train = items[0:mid]
        test = items[mid:size]
        return train, test

def count_ones(array1):
        size1 = np.shape(array1)[0]
        #size2 = np.shape(array2)[0]
        count1 = 0
        #count2 = 0
        for i in range(size1):
                if array1[i] == 1:
                        count1 = count1 + 1
        return count1/size1

def merge_arr_from_class(my_list):
        gender1 = []
        diag1 = []
        speaker_ID1 = []
        size = np.shape(my_list)[0]
        for i in range(size):
                gender1.append(my_list[i].gender[0])
                diag1.append(my_list[i].label[0])
                speaker_ID1.append(my_list[i].speaker_ID)
        
        #print(speaker_ID)
        return np.array(diag1), np.array(gender1), np.array(speaker_ID1)


def test_opt_problem():
        my_list = class_init()
        #for i in range(100):
        train, test = k_fold_cross_validation(my_list, 0.0, randomize = False)
        diag1, gender1, speaker_IDs1 = merge_arr_from_class(train)
        diag2, gender2, speaker_IDs2 = merge_arr_from_class(test)
        print(diag2.transpose(), "next \n", gender2.transpose())
               # L = abs(count_ones(diag1) - count_ones(diag2)) + abs(count_ones(gender1) - count_ones(gender2)) 
               # print(speaker_IDs1, speaker_IDs2, L)
        #print(np.shape(gender1)[0])
        #print(train_genders)
        #return 0

def ID2Index(id, ID):
        N = 67
        for i in range(N):
                if ID[i] == id:
                        return i
        return -1

def test_arr_correspond():
        my_list = class_init()
        diag, gender, ID = merge_arr_from_class(my_list)

        test1 = [72, 68, 45, 87, 84, 14, 83, 22, 32, 53, 56, 11, 24, 93, 8, 40, 67, 54, 34, 46, 4]
        train1 = [74, 17, 6, 61, 18, 57, 64, 27, 99, 12, 77, 90, 15, 30, 85, 52, 95, 7, 55, 66, 69, 79, 37, 60, 48, 21, 94, 82, 23, 2, 76, 96, 25, 65, 75, 70, 5, 13, 88, 42, 36, 92, 20, 35, 51, 43]

        test2 = [57, 82, 70, 45, 79, 27, 67, 36, 65, 60, 23, 69, 17, 24, 7, 94, 42, 64, 22, 18, 92]
        train2 = [61, 83, 74, 46, 51, 25, 56, 43, 75, 48, 52, 85, 77, 15, 32, 55, 40, 95, 13, 90, 11, 53, 54, 12, 72, 6, 20, 99, 68, 87, 4, 8, 96, 37, 34, 35, 2, 66, 76, 93, 21, 84, 5, 30, 14, 88]

        test3 = [77, 14, 46, 54, 84, 53, 94, 11, 69, 92, 30, 21, 66, 42, 60, 72, 27, 68, 5, 35, 65]
        train3 = [36, 37, 85, 70, 34, 8, 90, 17, 32, 82, 25, 23, 20, 61, 67, 12, 24, 87, 52, 40, 79, 64, 2, 96, 57, 18, 88, 56, 4, 15, 13, 22, 7, 83, 93, 6, 45, 75, 43, 51, 76, 48, 74, 95, 99, 55]

        test4 = [69, 37, 12, 18, 75, 14, 53, 2, 77, 21, 99, 65, 30, 42, 52, 94, 72, 95, 8, 64, 60]
        train4 = [74, 35, 87, 4, 5, 92, 88, 27, 90, 7, 66, 24, 82, 85, 48, 57, 56, 96, 32, 68, 45, 67, 25, 17, 93, 11, 6, 34, 15, 13, 43, 46, 22, 55, 61, 70, 20, 79, 84, 51, 36, 23, 40, 76, 83, 54]

        test5 = [32, 35, 61, 5, 7, 54, 79, 87, 11, 45, 64, 22, 77, 23, 82, 30, 21, 20, 70, 92, 53]
        train5 = [69, 17, 24, 65, 85, 88, 46, 55, 18, 68, 15, 84, 83, 51, 96, 25, 75, 43, 60, 37, 74, 56, 27, 12, 95, 4, 93, 36, 13, 67, 99, 57, 94, 42, 48, 14, 2, 34, 72, 8, 40, 52, 90, 6, 76, 66]
        
        size1 = np.shape(test4)[0]
        size2 = np.shape(train4)[0]
        index = []
        #print(np.shape(train3)[0], np.shape(train4)[0])
        
        for i in range(size2):
                a = ID2Index(train5[i],ID)
                index.append(a)
        print(index)

        #N = 20
        '''
        test1.sort()
        test2.sort()
        test3.sort()
        test4.sort()
        test5.sort()
        print(test1,"\n",test2,"\n",test3,"\n",test4,"\n",test5)

        test1 = [48, 45, 29, 58, 56, 9, 55, 15, 21, 34, 37, 6, 17, 62, 5, 26, 44, 35, 22, 30, 1]   
        train1 = [49, 11, 3, 40, 12, 38, 41, 19, 66, 7, 52, 60, 10, 20, 57, 33, 64, 4, 36, 43, 46, 53, 25, 39, 31, 14, 63, 54, 16, 0, 51, 65, 18, 42, 50, 47, 2, 8, 59, 27, 24, 61, 13, 23, 32, 28]

        test2 = [38, 54, 47, 29, 53, 19, 44, 24, 42, 39, 16, 46, 11, 17, 4, 63, 27, 41, 15, 12, 61]
        train2 = [40, 55, 49, 30, 32, 18, 37, 28, 50, 31, 33, 57, 52, 10, 21, 36, 26, 64, 8, 60, 6, 34, 35, 7, 48, 3, 13, 66, 45, 58, 1, 5, 65, 25, 22, 23, 0, 43, 51, 62, 14, 56, 2, 20, 9, 59]

        test3 = [52, 9, 30, 35, 56, 34, 63, 6, 46, 61, 20, 14, 43, 27, 39, 48, 19, 45, 2, 23, 42]
        train3 = [24, 25, 57, 47, 22, 5, 60, 11, 21, 54, 18, 16, 13, 40, 44, 7, 17, 58, 33, 26, 53, 41, 0, 65, 38, 12, 59, 37, 1, 10, 8, 15, 4, 55, 62, 3, 29, 50, 28, 32, 51, 31, 49, 64, 66, 36]

        test4 = [49, 23, 1, 2, 61, 59, 19, 60, 4, 43, 17, 54, 57, 31, 38, 37, 65, 21, 45, 29, 44]
        train4 = [49, 23, 58, 1, 2, 61, 59, 19, 60, 4, 43, 17, 54, 57, 31, 38, 37, 65, 21, 45, 29, 44, 18, 11, 62, 6, 3, 22, 10, 8, 28, 30, 15, 36, 40, 47, 13, 53, 56, 32, 24, 16, 26, 51, 55, 35]

        test5 = [21, 23, 40, 2, 4, 35, 53, 58, 6, 29, 41, 15, 52, 16, 54, 20, 14, 13, 47, 61, 34]
        train5 = [46, 11, 17, 42, 57, 59, 30, 36, 12, 45, 10, 56, 55, 32, 65, 18, 50, 28, 39, 25, 49, 37, 19, 7, 64, 1, 62, 24, 8, 44, 66, 38, 63, 27, 31, 9, 0, 22, 48, 5, 26, 33, 60, 3, 51, 43]
        '''
        #print(test1,"\n",test2,"\n",test3,"\n")
        
def test_validation():
        my_list = class_init()
        indexes = [53,65,4,62,61,63,25,37,64,46,23,26,13,55,39,57,5,3,41,11,38,12,34,16,31,56,29,51,60,8,28,18,33,54,42,14,19,40,36,6,58,49,32,7,27,9,59,21,43,67,2,45,17,30,15,35,10,24,52,47,44,50,22,1,66,48,20]
        out = k_fold_division(my_list,indexes)

        

        #print(np.shape(out[5])[0])

def read(csvPath):
    fileCVS = pd.read_csv(csvPath)
    return fileCVS
    
#def dim_elimination(Features, labels)
def reorder_array(my_list, indexes):
        out = []
        size = np.shape(my_list)[0]
        for i in range(size):
                indexes[i] = indexes[i] - 1
                out.append(my_list[indexes[i]])
        
        return out

# group_features(train_class, test_class) = np.array(train_features), np.array(train_labels), np.array(test_features), np.array(test_labels) 
# dim_elimination(Features, labels)
def main3():
        #tmp_list = reshape_class_init() #reshape is unimportant
        tmp_list = class_init()
        #indexes 0 to 5 all satisfy the best distribution on genders and types
        #indexes0 = [44,38,29,51,54,36,22,16,52,14,17,26,28,39,64,25,27,20,40,19,31,12,63,34,8,56,48,58,60,1,65,10,45,30,15,37,11,55,59,46,18,35,13,41,53,42,2,66,24,50,4,62,7,23,33,43,61,32,5,9,57,49,67,3,47,21,6]
        #indexes1 = [53,65,4,62,61,63,25,37,64,46,23,26,13,55,39,57,5,3,41,11,38,12,34,16,31,56,29,51,60,8,28,18,33,54,42,14,19,40,36,6,58,49,32,7,27,9,59,21,43,67,2,45,17,30,15,35,10,24,52,47,44,50,22,1,66,48,20]
        #indexes2 = [27,28,10,34,56,46,25,22,62,58,19,5,40,61,15,43,11,9,59,2,52,64,45,39,44,12,31,65,51,41,37,18,21,57,32,47,63,8,38,4,13,24,7,30,53,36,50,35,16,33,66,20,6,48,67,23,54,60,17,1,49,26,42,55,3,14,29]
        #indexes3 = [50,53,43,3,9,26,34,29,32,31,2,40,8,19,33,20,42,28,37,57,38,30,55,48,51,7,16,21,66,65,1,35,54,61,12,41,22,36,5,47,58,14,11,17,62,15,59,63,64,4,45,46,18,60,49,10,52,56,13,6,67,23,24,39,27,25,44]
        
        
        indexes4 = [54,38,10,40,1,46,67,39,29,13,2,16,11,43,65,41,25,21,59,24,49,3,37,62,15,64,9,20,6,55,31,30,34,8,53,44,63,4,19,60,14,26,36,58,17,23,48,35,12,47,42,22,5,18,45,27,52,28,33,7,56,66,32,57,51,50,61]

        indexes = indexes4
        my_list = reorder_array(tmp_list, indexes)

        #choose 5 folds only
        my_list = my_list[0:55]
        #test list from 56 to 66
        features, labels, null1, null2 = group_features(my_list, [])
        #print(np.shape(test_labels))
        ranking = dim_elimination(features, labels)
        print(ranking)
        '''
        for i in range(4):
                if(i == 0):
                        indexes = indexes1
                if(i == 1):
                        indexes = indexes2
                if(i == 2):
                        indexes = indexes3
                if(i == 3):
                        indexes = indexes4

                my_list = reorder_array(tmp_list, indexes)
                features, labels, null1, null2 = group_features(my_list, [])
                #print(np.shape(test_labels))
                ranking = dim_elimination(features, labels)
                print(ranking)
        '''
        '''
        issue: features are unequally spread out to different speakers, thus StraitKFold does not produce output in the
        desire distribution
        '''



def test4():
        #my_list = reshape_class_init()
        #print(my_list[5].features[0], np.shape(my_list[0].features[0])[0])
        #my_list1 = class_init()

        #out = select_best_features(my_list[0].features[0])
        #print(out, "\n\n\n", my_list[0].features[0])
        '''
        N = np.shape(my_list)[0]
        N_print = 16
        for i in range(N_print):
                print(my_list[44].features[i] , "\n",  my_list1[44].features[i])
                #if(my_list[44].features[i].all != my_list1[44].features[i].all):
                 #       print("wrong\n")
               #print((my_list[0].features[i]), "\n",  my_list1[0].features[i])
        '''

#initialize()
#test()
#main()
#test1()
#class_init()
#main2()
#test_split()
#test_arr_correspond()
#test_opt_problem()
#test_validation()
main2()
#main3()
#test4()