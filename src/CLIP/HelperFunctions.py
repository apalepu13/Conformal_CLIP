import regex as re
from Data_Loading import *
import torch
import torch.nn as nn
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getDatasets(subset = ['train', 'val', 'test'], augs = 1,
                heads = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
                filters = [], frontal = False, lateral = False):
    '''
    Gets particular subsets of the MIMIC_CXR data
    :param subset: which data subsets to extract  :param augs: number of im augs (need to update fns)
    :param heads: which classification labels to extract
    :param filters: which filters to use for images
    :param frontal:get only frontal  :param lateral: get only lateral
    '''

    if frontal:
        filters += ['frontal']
    elif lateral:
        filters += ['lateral']
    if type(subset) == str:
        subset = [subset]

    datlist = {}
    for sub in subset:
        datlist[sub] = Image_Text_Dataset(group=sub, out_heads = heads, im_aug = augs, filters = filters)
    return datlist

def getLoaders(datasets, args=None, subset = ['train', 'val', 'test'], num_work = 16, shuffle=True):
    '''
    Gets Dataloaders for datasets
    :param datasets returned by getDatasets:
    :param args args.batch_size to specify batch size:
    '''
    loaders = []
    for sub in subset:
        loaders.append(DataLoader(datasets[sub], batch_size=args.batch_size if args else 32, shuffle=shuffle, num_workers=num_work,
                                  prefetch_factor=max(1, int(args.batch_size/num_work)) if args.batch_size else 1, pin_memory=True))
    return loaders

def getExperiment(args, mp):
    '''
    Returns filepath to new experiment (or current if args.resume == True)
    '''
    if args.debug:
        return "debug"

    if not os.listdir(os.path.join(mp)):
        print("No models exist, creating directory")
        fp = os.path.join(mp, 'exp1')
    elif os.listdir(os.path.join(mp)):
        all_files = os.listdir(os.path.join(mp))
        je_exps = [exp for exp in all_files if 'exp' in exp]
        num = [int(re.search('\d+', exp).group(0)) for exp in je_exps]
        highest_ind = np.argmax(np.array(num))
        highest = num[highest_ind]
        if not args.resume:
            highest = highest + 1
        fp = os.path.join(mp, 'exp'+str(highest))

    return fp

def startExperiment(args, je_model, optimizer, fp):
    '''
    Loads current experiment or creates new experiment
    je_model, optimizer is the architecture,optimizer to load weights into
    fp is determined by getExperiment
    '''
    if fp == "debug":
        return 0, 100000, args

    if args.resume:
        if os.listdir(os.path.join(fp)):
            all_files = os.listdir(os.path.join(fp))
            je_files = [file for file in all_files if 'model' in file]
            num = [int(re.search('\d+', file).group(0)) for file in je_files]
            highest = np.argmax(np.array(num))
            loadpath = os.path.join(fp, np.array(je_files)[highest])
            print("Loading " + loadpath)
            checkpoint = torch.load(loadpath)
            je_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            args = checkpoint['args']
        else:
            raise Exception("Experiment doesn't exist")
    else:
        print("Starting from scratch")
        start = 0
        best_val_loss = 1000000000
        if not args.debug:
            os.makedirs(fp)
            txt = args.desc
            with open(os.path.join(fp, "desc.txt"), "w") as text_file:
                text_file.write(txt)
    return start, best_val_loss, args

def clip_loss(im_logits, text_logits, loss_weight = 1, criterion = nn.CrossEntropyLoss()):
    '''
    Return loss using cosine similarities of batch of im/text embeddings (im_logits, text_logits)
    loss_weight = scalar to multiply loss by
    '''
    samp = torch.tensor(np.arange(im_logits.shape[0]))
    loss_a = criterion(im_logits, samp.to(device)) #1
    loss_b = criterion(text_logits, samp.to(device)) #1
    closs = (loss_a + loss_b) / 2
    closs = closs * loss_weight #1
    return closs #1

def train(je_model, im1, im2, texts, tokenizer):
    '''
    Forward pass for batch of (im1,im2,texts) using je_model
    '''
    je_model.zero_grad(set_to_none=True)
    # Set mini-batch dataset
    images1 = im1.to(device)
    images2 = im2.to(device)
    texts = tokenizer.encode(texts=texts).to(device)

    # Forward, backward and optimize
    im_logits1, text_logits1 = je_model(images1, texts)
    im_logits2, text_logits2 = je_model(images2, texts)
    cl1 = clip_loss(im_logits1, text_logits1, device)
    cl2 = clip_loss(im_logits2, text_logits2, device)
    iloss = clip_loss(im_logits1, im_logits2, device)
    loss = cl1 + cl2 + iloss
    return loss

def validate(val_data_loader, tokenizer, je_model, proportion = 1.0):
    '''
    Forward pass for entire val dataset, returning avg val loss
    '''
    vlosses = []
    with torch.no_grad():
        for j, (valims1, valims2, valDFs, valtexts) in enumerate(val_data_loader):
            gen = np.random.rand(1)
            if gen >= proportion:
                continue

            valims1 = valims1.to(device)
            valims2 = valims2.to(device)
            valtexts = tokenizer.encode(texts=valtexts).to(device)
            val_im1, val_t1 = je_model(valims1, valtexts)
            val_im2, val_t2 = je_model(valims2, valtexts)
            myloss = clip_loss(val_im1, val_t1)
            myloss += clip_loss(val_im2, val_t2)
            myloss += clip_loss(val_im1, val_im2)
            vlosses.append(myloss.cpu())

    vlloss = np.mean(np.array(vlosses))
    print(' Val Loss: ' + str(vlloss))
    return vlloss

def b_loss(impreds, labels, heads, criterion):
    '''
    binary cross entropy loss for classifier predictions
    impreds = prediction logits; N by len(heads)
    labels = labels: N by len(heads)
    heads = list of labels
    '''
    losses = torch.zeros(len(heads))
    labels = getLabels(labels, heads)
    for i, h in enumerate(heads):
        label = labels[:, i]
        mypreds = impreds[torch.logical_not(torch.isnan(label)), i]
        mylabels = label[torch.logical_not(torch.isnan(label))]
        losses[i] = criterion(mypreds, mylabels)
    losses = losses[torch.logical_not(torch.isnan(losses))]
    loss = torch.mean(losses)
    if torch.isnan(loss):
        loss = 0
    return loss

def train_vision(vision_model, im1, im2, labels, heads, criterion=torch.nn.BCEWithLogitsLoss()):
    '''
    Forward pass for classifier training for batch of inputs/outputs
    vision_model- classifier model, im1, im2= image inputs
    label- dictionary of labels, heads = label keys
    '''
    vision_model.zero_grad(set_to_none=True)
    images1 = im1.to(device)
    images2 = im2.to(device)

    # Forward, backward and optimize
    impreds1 = vision_model(images1)
    cl1 = b_loss(impreds1, labels, device, heads, criterion)
    impreds2 = vision_model(images2)
    cl2 = b_loss(impreds2, labels, device, heads, criterion)
    loss = cl1 + cl2
    return loss

def validate_vision(val_data_loader, vision_model, heads, criterion, proportion = 1.0):
    '''
    Forward pass for full val dataset for image classifier
    Returns val loss
    '''
    vlosses = []
    with torch.no_grad():
        for j, res in enumerate(val_data_loader):
            gen = np.random.rand(1)
            if gen >= proportion:
                continue
            valims1, valims2, val_labels, val_text = res
            valims1 = valims1.to(device)
            valpred1= vision_model(valims1)
            myloss = b_loss(valpred1, val_labels, device, heads, criterion)
            valims2 = valims2.to(device)
            valpred2 = vision_model(valims2)
            myloss = myloss + b_loss(valpred2, val_labels, device, heads, criterion)
            vlosses.append(myloss.cpu())

    vlloss = np.mean(np.array(vlosses))
    print('Val Loss: ' + str(vlloss))
    return vlloss

def getLabels(df, heads):
    '''
    Gets the labels specified by list heads from dictionary df into a tensor and returns it
    '''
    labels = None
    for i, h in enumerate(heads):
        label = df[h].float()
        label[label == -1.0] = float('nan')
        label[label == 0.0] = 0.0
        label[label == 1.0] = 1.0
        if labels is None:
            labels = label
            labels = labels[:, None]
        else:
            labels = torch.cat((labels, label[:, None]), axis=1)

    return labels

def get_all_preds(DL, mod, heads = ['covid19', 'No Finding'], device='cuda'):
    '''
    From a given dataloader, returns a full list of predictions and labels using a given model "mod"
    '''
    tp, tt = None, None
    for i, res in enumerate(DL):
        try:
            im1, im2, df, study = res
        except:
            im1, df = res

        images = im1.to(device)
        preds = mod(images).to(device)
        labels = getLabels(df, heads).to(device)

        if tp is None:
            tp = preds
        else:
            tp = torch.cat((tp, preds), axis=0)
        if tt is None:
            tt = labels
        else:
            tt = torch.cat((tt, labels), axis=0)

    return tp, tt


