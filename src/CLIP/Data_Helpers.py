import pandas as pd
import numpy as np
import re
from sklearn.model_selection import GroupShuffleSplit

def getFilters(exp_path, overwrite = '', toprint=True):
    '''
    return filters that were used to train an experiment, if possible
    Looking for 'filters.txt' in the exp folder.
    Alternatively, can overwrite the filters used
    '''
    try:
        if overwrite is not '' and type(overwrite) == str:
            if toprint:
                print("Overwriting filters with " + overwrite)
            return overwrite.split(",")
        else:
            txt_file = open(exp_path + 'filters.txt', "r")
            file_content = txt_file.read()
            content_list = file_content.split(",")
            txt_file.close()
            if toprint:
                print("Using filter file with " + file_content)
            return content_list
    except:
        if toprint:
            print("No filter file found, none applied.")
        return []

def getImList(group, fps, filters = []):
    '''
    Returns the dataframe of samples (il) to use and the root directory for the data
    '''
    rd = fps['mimic_root_dir']
    il_labels = pd.read_csv(fps['mimic_chex_file'])
    il_meta = pd.read_csv(fps['mimic_meta_file']).loc[:, ['dicom_id', 'subject_id', 'study_id', 'ViewPosition']]
    il = pd.read_csv(fps['mimic_csv_file'])
    il = il.merge(il_labels, on=['subject_id', 'study_id'])
    il = il.merge(il_meta, on=['dicom_id','subject_id', 'study_id'])
    if 'frontal' in filters:
        il = il[np.logical_or(il['ViewPosition'] == 'AP', il['ViewPosition'] == 'PA')]
    elif 'lateral' in filters:
        il = il[np.logical_not(np.logical_or(il['ViewPosition'] == 'AP', il['ViewPosition'] == 'PA'))]

    il = getSplits(il, group)
    il['pGroup'] = np.array(["p" + pg[:2] for pg in il['subject_id'].values.astype(str)])
    il['pName'] = np.array(["p" + pn for pn in il['subject_id'].values.astype(str)])
    il['sName'] = np.array(["s" + sn for sn in il['study_id'].values.astype(str)])
    return il, rd


def getSplits(df, group):
    '''
    Returns the data for a specified group (train/val/calib/test)
    '''
    if 'tiny' in group:
        df = splitDF(df, 'subject_id', 0.01)[1]

    trainval, calibtest = splitDF(df, 'subject_id', 0.2)
    train, val = splitDF(trainval, 'subject_id', 0.1)
    calib, test = splitDF(calibtest, 'subject_id', 0.5)  # train, val, calib, test

    if 'train' in group:
        df = train
    elif 'val' in group:
        df = val
    elif 'calib' in group:
        df = calib
    else:
        df = test

    return df

def splitDF(df, patientID, testsize=0.2):
    '''
    Splitting data with given test size with all data from a given patient in one group
    '''
    splitter = GroupShuffleSplit(test_size=testsize, n_splits=1, random_state=1)
    split = splitter.split(df, groups=df[patientID])
    train_inds, valtest_inds = next(split)
    train = df.iloc[train_inds]
    test = df.iloc[valtest_inds]
    return train, test

def textProcess(text):
    '''
    Code to extract relevant portion of clinical reports with regex
    Currently, trying to extract findings and impression sections
    '''
    sections = 'WET READ:|FINAL REPORT |INDICATION:|HISTORY:|STUDY:|COMPARISONS:|COMPARISON:|TECHNIQUE:|FINDINGS:|IMPRESSION:|NOTIFICATION:'
    mydict = {}
    labs = re.findall(sections, text)
    splits = re.split(sections, text)
    for i, l in enumerate(splits):
        if i == 0:
            continue
        else:
            if len(splits[i]) > 50 or labs[i - 1] == 'IMPRESSION:':
                mydict[labs[i - 1]] = splits[i]

    if 'FINDINGS:' in mydict.keys():
        if 'IMPRESSION:' in mydict.keys():
            mystr = "FINDINGS: " + mydict['FINDINGS:'] + "IMPRESSION: " + mydict['IMPRESSION:']
        else:
            mystr = "FINDINGS: " + mydict['FINDINGS:']
    else:
        mystr = ""
        if 'COMPARISONS:' in mydict.keys():
            mystr = mystr + "COMPARISONS: " + mydict['COMPARISONS:']
        if 'COMPARISON:' in mydict.keys():
            mystr = mystr + "COMPARISONS: " + mydict['COMPARISON:']
        if 'IMPRESSION:' in mydict.keys():
            mystr = mystr + "IMPRESSION: " + mydict['IMPRESSION:']
    if len(mystr) > 100:
        return mystr
    else:
        if 'FINAL REPORT ' in mydict.keys() and len(mydict['FINAL REPORT ']) > 100:
            return mydict['FINAL REPORT '] + mystr
        else:
            return text