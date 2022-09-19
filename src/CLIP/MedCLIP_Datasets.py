import torch
import os
import numpy as np
from PIL import Image
import MedDataHelpers
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MedDataset(Dataset):
    """Chx - Report dataset.""" #d
    def __init__(self, source = 'mimic_cxr', group='train', im_aug = 1, train_mode=False,
                 out_heads = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
                 filters = []):
        # filepaths
        fps = {}
        fps['mimic_root_dir'] = '/n/data2/hms/dbmi/beamlab/mimic_cxr/'
        fps['mimic_meta_file'] = '/n/data2/hms/dbmi/beamlab/mimic_cxr/mimic-cxr-2.0.0-metadata.csv'
        fps['mimic_csv_file'] = '/n/data2/hms/dbmi/beamlab/mimic_cxr/mimic-cxr-2.0.0-split.csv'
        fps['mimic_chex_file'] = '/n/data2/hms/dbmi/beamlab/mimic_cxr/mimic-cxr-2.0.0-chexpert.csv'
        fps['indiana_csv_file'] = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/indiana_cxr_list.csv'
        fps['indiana_root_dir'] = '/n/data2/hms/dbmi/beamlab/indiana_cxr/'
        fps['chexpert_root_dir'] = '/n/data2/hms/dbmi/beamlab/chexpert/'
        fps['covid_chestxray_csv_file'] = '/n/data2/hms/dbmi/beamlab/covid19qu/infection_dat/data_list.csv'
        fps['medpix_root_dir'] = '/n/data2/hms/dbmi/beamlab/medpix/'
        fps['medpix_file'] = 'medpix_case_data_table.csv'
        sourceDict = {'m': 'mimic_cxr', 'ms': 'ms_cxr', 'i': 'indiana_cxr', 'c': 'chexpert', 'co': 'covid-chestxray', 'med': 'medpix'}

        #setting attributes
        self.heads = out_heads
        self.source = (sourceDict[source] if source in sourceDict.keys() else source)
        self.group = group
        self.im_aug = im_aug
        self.train_mode = train_mode

        # Image processing
        self.im_preprocessing_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, ratio=(.9, 1.0)),
            transforms.RandomAffine(10, translate=(.05, .05), scale=(.95, 1.05)),
            transforms.ColorJitter(brightness=.2, contrast=.2),
            #transforms.GaussianBlur(kernel_size=15, sigma=(.1, 3.0))
            transforms.Resize(size=(224, 224))])

        self.im_preprocessing_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)])

        self.im_finish = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.totensor = transforms.Compose([
            transforms.ToTensor()
        ])

        self.im_list, self.root_dir = MedDataHelpers.getImList(sr = self.source, group = self.group, fps = fps,
                                                heads = self.heads, filters = filters)
        print(self.source, group + " size= " + str(self.im_list.shape))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        sample = {}
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.source == 'mimic_cxr' or self.source == 'mscxr':
            ims = self.im_list.iloc[idx, :]
            img_name = os.path.join(self.root_dir, ims['pGroup'], ims['pName'], ims['sName'], ims['dicom_id'] + '.jpg')
            image = Image.open(img_name).convert("RGB")
            df = ims.loc[self.heads]
            if self.group == 'test' or self.source == 'mscxr' or not self.train_mode:
                images = [self.im_preprocessing_test(image) for i in range(self.im_aug)]
            else:
                images = [self.im_preprocessing_train(image) for i in range(self.im_aug)]

            images = [self.im_finish(im) for im in images]
            text = ims['text']
            sample['images'] = images
            sample['labels'] = df.to_dict()
            sample['texts'] = text #ims, df, text
            return sample

        elif self.source == 'indiana_cxr':
            imstexts = self.im_list.iloc[idx, :]
            ims = imstexts.loc['images']
            ims = ims[:-4] + '.png'
            img_name = os.path.join(self.root_dir, 'indiana_cxr', ims)
            image = Image.open(img_name)
            image = image.convert("RGB")
            images = [self.im_preprocessing_train(image) for i in range(self.im_aug)]
            images = [self.im_finish(im) for im in images]
            text = imstexts.loc['texts']
            sample['images'] = images
            sample['texts'] = text  # ims, df, text
            return sample

        elif self.source == 'chexpert':
            df = self.im_list.iloc[idx, :]
            img_name = self.root_dir + df['Path']
            image = Image.open(img_name)
            image = image.convert("RGB")
            if self.group == 'test' or self.group == 'all':
                image = self.im_preprocessing_test(image)
            else:
                image = self.im_preprocessing_train(image)

            image = self.im_finish(image)
            df = df.loc[self.heads]
            sample['images'] = [image]
            sample['labels'] = df.to_dict()
            return sample #ims, df

        elif self.source == 'covid-chestxray':
            df = self.im_list.iloc[idx, :]
            img_name = df['im_path']
            image = Image.open(img_name)
            image = image.convert("RGB")
            image = self.im_preprocessing_test(image)
            image = self.im_finish(image)
            lung_mask = Image.open(df['lung_path'])
            inf_mask = Image.open(df['inf_path'])
            lung_mask = self.totensor(lung_mask)
            inf_mask = self.totensor(inf_mask)
            df = df.loc[['No Finding', 'Pneumonia', 'covid19']]
            sample['images'] = [image]
            sample['labels'] = df.to_dict()
            sample['inf_mask'] = inf_mask
            sample['lung_mask'] = lung_mask

if __name__=='__main__':
    train_dat = MedDataset(source = 'mimic_cxr', group = 'train', im_aug=3)
    for i in np.arange(3333,3336):
        result = train_dat.__getitem__(i)
        print(result['labels'])

