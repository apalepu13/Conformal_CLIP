import copy
import os
import pickle
import time

import CLIP_Embedding
import MedCLIP_Datasets
import MedDataHelpers
import PIL
import Vision_Model
import matplotlib.pyplot as plt
import modules
import numpy as np
import pandas as pd
import resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import train
import utils
from torchvision import datasets, models, transforms
from tqdm.notebook import tqdm

os.chdir('/n/data2/hms/dbmi/beamlab/bhawesh/Conformal_CLIP_main/src/CLIP')

path_to_model = "/n/data2/hms/dbmi/beamlab/bhawesh/Conformal_CLIP/models/CLIP_model/exp1/best_model.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CLIP_Embedding.MedCLIP().to(device)
checkpoint = torch.load(path_to_model, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

datasets = MedDataHelpers.getDatasets('mimic_cxr', subset=['train', 'val', 'calib', 'test'], filters=['frontal'])
batch_size = 32
dl = MedDataHelpers.getLoaders(datasets, batchsize=batch_size)


def get_embeddings(datagen, max_i=180000, batch_size=32):
    img_embed_list = []
    text_embed_list = []
    label_list = []
    k = 0
    total_iter = max_i/batch_size
    for i, samples in enumerate(datagen):
        im_embs, text_embs = CLIP_Embedding.getEmbeddings(model, samples)
        img_embed_list.append(im_embs.cpu().detach().numpy())
        text_embed_list.append(text_embs.cpu().detach().numpy())
        label_list.append(samples['labels'])
        del im_embs, text_embs
        if (k*total_iter//(100) == i):
            k+=1
            print("Completed "+str(k-1)+"%") 
        if i == (max_i // batch_size):
            break
    img_embs = np.concatenate(img_embed_list, axis=0)
    text_embs = np.concatenate(text_embed_list, axis=0)
    return {"img": img_embs, "text": text_embs, "labels": label_list}


dict_train_embeddings = get_embeddings(dl['train'],max_i = 1000,  batch_size=batch_size)

with open('dict_embeddings_train_complete.pkl', 'wb') as f:
    pickle.dump(dict_train_embeddings, f)
