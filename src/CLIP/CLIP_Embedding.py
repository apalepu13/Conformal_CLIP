from torch import nn
import torch
import Vision_Model
from MedDataHelpers import *
from transformers import AutoTokenizer, AutoModel
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MedCLIP(nn.Module):
    def __init__(self, freeze_transformer=True):
        super().__init__()
        url = "microsoft/BiomedVLP-CXR-BERT-specialized"
        self.cnn = Vision_Model.get_biovil_resnet()
        self.tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
        self.transformer = AutoModel.from_pretrained(url, trust_remote_code=True)

        for param in self.transformer.parameters():
            param.requires_grad = True
        if freeze_transformer:
            modules = [self.transformer.bert.embeddings, *self.transformer.bert.encoder.layer[:8]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def similarity_matrix(self, emb1, emb2): #N E, N E
        image_features = emb1 / emb1.norm(dim=-1, keepdim=True)  # N E
        text_features = emb2 / emb2.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        return logits_per_image

    def forward(self, images, text): #k images
        if not isinstance(images, list):
            images = [images]
        if not isinstance(text, list):
            text = [text]

        images = [im[None, :] if im.dim() == 3 else im for im in images]
        images = [im.to(device) for im in images]

        token_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                               add_special_tokens=True, truncation=True,
                                               padding='longest', max_length=256,
                                               return_tensors='pt').to(device)
        text_emb = self.transformer.get_projected_text_embeddings(input_ids=token_output.input_ids,
                                                         attention_mask=token_output.attention_mask).to(device)
        all_im_embs, im_logits = [], []
        for im in images:
            output = self.cnn(im)
            image_emb = output.projected_global_embedding #N E
            all_im_embs.append(image_emb)
            im_logits.append(self.similarity_matrix(image_emb, text_emb))

        aug_logits = None
        if len(images) > 1:
            aug_logits = []
            for i in np.arange(len(images)):
                for j in np.arange(len(images)):
                    if i <= j:
                        continue
                    imsims = self.similarity_matrix(all_im_embs[i], all_im_embs[j])
                    aug_logits.append(imsims)
        return im_logits, aug_logits #list per im,, #list per im-im pair

def getEmbeddings(model, samples):
    images = samples['images']
    texts = samples['texts']
    token_output = model.tokenizer.batch_encode_plus(batch_text_or_text_pairs=texts,
                                               add_special_tokens=True, truncation=True,
                                               padding='longest', max_length=256,
                                               return_tensors='pt').to(device)

    text_embs = model.transformer.get_projected_text_embeddings(input_ids=token_output.input_ids,
                                                                  attention_mask=token_output.attention_mask).to(device)
    all_im_embs = []
    if len(images) == 1:
        return model.cnn(images[0].to(device)).projected_global_embedding, text_embs

    for j, image in enumerate(images):
        im_embs = model.cnn(image.to(device)).projected_global_embedding
        all_im_embs.append(im_embs)

    return all_im_embs, text_embs #list of image embeddings, text embeddings


if __name__=='__main__':
    model = MedCLIP().to(device)
    checkpoint = torch.load('/n/data2/hms/dbmi/beamlab/anil/Conformal_CLIP/models/CLIP_model/exp2/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    datasets = getDatasets('mimic_cxr', subset=['tinytrain', 'tinyval', 'tinycalib', 'tinytest'], filters=['frontal'], augs=1)
    dl = getLoaders(datasets)
    for i, samples in enumerate(dl['tinycalib']):
        im_embs, text_embs = getEmbeddings(model, samples)
        print(im_embs.shape, text_embs.shape)