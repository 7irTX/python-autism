import numpy as np
from Diagnosis_class import Diagnosis_class
from main import read
import config as cf

def class_init():
        tmp_list = []
        path_label = "C:/Users/Yitian Chen.DESKTOP-TIIP443/Desktop/tmp1/training/result_csv/all_label.csv"
        path_result = "C:/Users/Yitian Chen.DESKTOP-TIIP443/Desktop/tmp1/training/result_csv/all_result.csv"
        path_speakerID = "C:/Users/Yitian Chen.DESKTOP-TIIP443/Desktop/UNSW_Thesis/ComParE2013_Autism/lab/ComParE2013_Autism.csv"
        tmp_label = read(path_label)
        tmp_result = read(path_result)
        tmp_speaker = read(path_speakerID)
        label = np.array(tmp_label)
        features = np.array(tmp_result)
        speakerID = np.array(tmp_speaker)[:,1]
        N1 = np.shape(label)[0]
        N2 = cf.NUM_SPEAKERS
        for i in range(N1):
                tmp_list.append(Diagnosis_class(label[i],speakerID[i],features[i]))
        tmp1 = np.array(tmp_list[0].features)
        tmp2 = np.array(tmp_list[1].features)
        tmp_list[0].features = np.row_stack((tmp1,tmp2))
        #print(tmp_list[0].features, tmp_list[0].speaker_ID)
        
        '''
        for i in range(N1):
                sort_array.append(tmp_list[i].speaker_ID)
        '''