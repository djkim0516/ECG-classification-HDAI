import os, glob
import array
import json
import base64
import xmltodict
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# %matplotlib inline

def xml_to_DataFrame(xml_file_dir):
    LOCAL_PATH = os.path.join(xml_file_dir)
    with open(LOCAL_PATH, 'rb') as xml:
        ecg = xmltodict.parse(xml.read().decode('iso-8859-1'))
    total_column_list = ['median_I', 'median_II', 'median_III', 'median_aVR', 'median_aVL', 'median_aVF', 'median_V1', 'median_V2', 'median_V3', 'median_V4', 'median_V5', 'median_V6','rhythm_I', 'rhythm_II','rhythm_III', 'rhythm_aVR', 'rhythm_aVL', 'rhythm_aVF', 'rhythm_V1', 'rhythm_V2', 'rhythm_V3', 'rhythm_V4', 'rhythm_V5', 'rhythm_V6']
    total_result = pd.DataFrame(columns=total_column_list, index=[i for i in range(5000)])
    try:
        try:
            column_list = ['median_I', 'median_II', 'median_V1', 'median_V2', 'median_V3', 'median_V4', 'median_V5', 'median_V6','rhythm_I', 'rhythm_II', 'rhythm_V1', 'rhythm_V2', 'rhythm_V3', 'rhythm_V4', 'rhythm_V5', 'rhythm_V6']
            result = pd.DataFrame(columns=column_list, index=[i for i in range(5000)])
            for i in range(8):
                lead_b64_median = base64.b64decode(ecg['RestingECG']['Waveform'][0]['LeadData'][i]['WaveFormData'])
                lead_vals_median = np.array(array.array('h', lead_b64_median))
                result.iloc[:len(lead_vals_median),i] = lead_vals_median    
                lead_b64_rhythm = base64.b64decode(ecg['RestingECG']['Waveform'][1]['LeadData'][i]['WaveFormData'])
                lead_vals_rhythm = np.array(array.array('h', lead_b64_rhythm))
                result.iloc[:len(lead_vals_rhythm),i+8] = lead_vals_rhythm    
        except:
            if len(ecg['RestingECG']['Waveform']['LeadData']) == 8:
                column_list = ['median_I', 'median_II', 'median_V1', 'median_V2', 'median_V3', 'median_V4', 'median_V5', 'median_V6','rhythm_I', 'rhythm_II', 'rhythm_V1', 'rhythm_V2', 'rhythm_V3', 'rhythm_V4', 'rhythm_V5', 'rhythm_V6']
                result = pd.DataFrame(columns=column_list, index=[i for i in range(5000)])
                for i in range(8):    
                    lead_b64_rhythm = base64.b64decode(ecg['RestingECG']['Waveform']['LeadData'][i]['WaveFormData'])
                    lead_vals_rhythm = np.array(array.array('h', lead_b64_rhythm))
                    result.iloc[:len(lead_vals_rhythm),i+8] = lead_vals_rhythm  
            else:          
                column_list = ['median_I', 'median_II', 'median_III', 'median_aVR', 'median_aVL', 'median_aVF', 'median_V1', 'median_V2', 'median_V3', 'median_V4', 'median_V5', 'median_V6','rhythm_I', 'rhythm_II','rhythm_III', 'rhythm_aVR', 'rhythm_aVL', 'rhythm_aVF', 'rhythm_V1', 'rhythm_V2', 'rhythm_V3', 'rhythm_V4', 'rhythm_V5', 'rhythm_V6']
                result = pd.DataFrame(columns=column_list, index=[i for i in range(5000)])      
                for i in range(12):    
                    lead_b64_rhythm = base64.b64decode(ecg['RestingECG']['Waveform']['LeadData'][i]['WaveFormData'])
                    lead_vals_rhythm = np.array(array.array('h', lead_b64_rhythm))
                    result.iloc[:len(lead_vals_rhythm),i+12] = lead_vals_rhythm
        for col in column_list:
            total_result.loc[:,col] = result.loc[:,col]    
    except:
        pass
    
    return total_result

def json_get_lbl(json_filename):
    lbl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(json_filename)))), 'label', 
                           os.path.basename(os.path.dirname(os.path.dirname(json_filename))), os.path.basename(os.path.dirname(json_filename)), os.path.basename(json_filename))
    with open(lbl_dir[:-4]+'.json') as jsonfile:
        lbl = json.load(jsonfile)
    lbl = lbl['labels'][0]['label_id']
    return lbl




#전처리 데이터 저장할 directory 생성
'''
if not os.path.exists("./electrocardiogram/data_denoised"):
    os.makedirs("./electrocardiogram/data_denoised/train/arrhythmia")
    os.makedirs("./electrocardiogram/data_denoised/train/normal")
    os.makedirs("./electrocardiogram/data_denoised/validation/arrhythmia")
    os.makedirs("./electrocardiogram/data_denoised/validation/normal")
    os.makedirs("./electrocardiogram/data_denoised/test/arrhythmia")  #--- test시 생성
    os.makedirs("./electrocardiogram/data_denoised/test/normal")      #--- test시 생성

#notch filter
fs = 500.0      #sampling freq.
f0 = 60.0
Q = 30.0
notch_b, notch_a = signal.iirnotch(f0, Q, fs)   #notch filter to remove 60Hz noise
'''

#bandpass filter
fs = 500.0
fb = 0.67
fh = 15
high_b = signal.firwin(201, [fb, fh], fs=fs, pass_zero='bandpass')      #bandpass filter for freq 0.67 ~ 15

#rhythm column 만 사용
total_column_list = ['rhythm_I', 'rhythm_II','rhythm_V1', 'rhythm_V2', 'rhythm_V3', 'rhythm_V4', 'rhythm_V5', 'rhythm_V6']
#'rhythm_III', 'rhythm_aVR', 'rhythm_aVL', 'rhythm_aVF', 
# DATA_DIR = glob.glob(os.path.join(os.path.relpath(path='./electrocardiogram'), 'data', '*', '*'))      #test data는 뒤에서 두번째 * 에 test 입력!!!!


def data_lbl_to_array(DATA_DIR):
    
    file_dir_each_data = list(list() for _ in range(len(DATA_DIR)))
    dir_idx = 0
    for file_dir in DATA_DIR:
        
        totaldata = np.empty(shape = (0,40001))     #전체 저장할 numpy array
        
        files = glob.glob(os.path.join(file_dir, "*.xml"))
        
        for file in files:   
            single_data = xml_to_DataFrame(file)
            single_data = single_data[total_column_list]  #only rhythms
            col_val_count = list(single_data.count())
            idx = 0
            for count in col_val_count:
                if count != 5000 and count != 0:
                    tmp_col = single_data.iloc[:, idx]
                    tmp_col.ffill(inplace=True)             #마지막행 데이터 없는 column ffill
                    single_data.iloc[:, idx] = tmp_col
                idx += 1

            ### noise 제거
            # single_data = signal.lfilter(notch_b, notch_a, single_data, axis=0)
            single_data = signal.lfilter(high_b, [1.0], single_data, axis=0)
            single_data = pd.DataFrame(single_data, columns=total_column_list)
            
            single_data = single_data.round(decimals=1)       #round to 1 decimal
            
            # lbl = json_get_lbl(os.path.join(os.path.dirname(file), file[-18:-4]))    #file directory중 train 부터 입력
            lbl = json_get_lbl(os.path.join(os.path.dirname(file), os.path.basename(file)))    #file directory중 train 부터 입력
            
            single_data = single_data.to_numpy().reshape((1,40000))
            lbl = np.array(lbl).reshape(1,1)
            
            data_lbl = np.concatenate((single_data, lbl), axis=-1)
            
            totaldata = np.concatenate((totaldata, data_lbl), axis=0)
            
        file_dir_each_data[dir_idx] = totaldata
        dir_idx += 1
            
    return file_dir_each_data