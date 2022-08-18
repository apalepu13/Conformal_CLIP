import torch
import os
from PIL import Image
from Data_Helpers import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class Image_Text_Dataset(Dataset):
    '''
    Specifies the dataset class for MIMIC_CXR images, texts and labels
    '''
    def __init__(self, group='train', im_aug = 2,
                 out_heads = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'], filters = []):
        '''
        group = which subset to extract from mimic_cxr
        im_aug = number of image augmentations
        out_heads = which labels to extract
        filters = which subset of images to include. [] for all, ['frontal'] & ['lateral'] are other options
        '''
        # filepaths
        fps = {}
        fps['mimic_root_dir'] = '/n/data2/hms/dbmi/beamlab/mimic_cxr/'
        fps['mimic_meta_file'] = '/n/data2/hms/dbmi/beamlab/mimic_cxr/mimic-cxr-2.0.0-metadata.csv'
        fps['mimic_csv_file'] = '/n/data2/hms/dbmi/beamlab/mimic_cxr/mimic-cxr-2.0.0-split.csv'
        fps['mimic_chex_file'] = '/n/data2/hms/dbmi/beamlab/mimic_cxr/mimic-cxr-2.0.0-chexpert.csv'

        #setting attributes
        self.heads = out_heads
        self.group = group
        self.im_aug = im_aug

        # Image processing
        self.im_preprocessing_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, ratio=(.8, 1.0)),
            transforms.RandomAffine(20, translate=(.1, .1), scale=(.95, 1.05)),
            transforms.ColorJitter(brightness=.4, contrast=.4),
            transforms.GaussianBlur(kernel_size=15, sigma=(.2, 2.5)),
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

        self.im_list, self.root_dir = getImList(group = self.group, fps = fps, filters = filters)
        print("MIMIC CXR", group + " size= " + str(self.im_list.shape))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ims = self.im_list.iloc[idx, :]
        img_name = os.path.join(self.root_dir, ims['pGroup'], ims['pName'], ims['sName'], ims['dicom_id'] + '.jpg')
        image = Image.open(img_name).convert("RGB")
        df = ims.loc[self.heads]
        if 'test' in self.group or 'calib' in self.group:
            images = [self.im_preprocessing_test(image)]
        else:
            images = [self.im_preprocessing_train(image) for i in range(self.im_aug)]

        images = [self.im_finish(im) for im in images]
        sample = images
        sample = sample + [df.to_dict()]
        text_name = os.path.join(self.root_dir, ims['pGroup'], ims['pName'], ims['sName'], ims['sName'] + '.txt')
        with open(text_name, "r") as text_file:
            text = text_file.read()
        text = text.replace('\n', '')
        text = textProcess(text)
        sample = sample + [text]
        return sample #image(s), labels, text

if __name__=='__main__':
    train_dat = Image_Text_Dataset(source = 'mimic_cxr', group = 'conformal_train', synth=False, im_aug=3)
    for i in np.arange(3333,3334):
        result = train_dat.__getitem__(i)

