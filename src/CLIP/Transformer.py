import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getLabelEmbeddings(mycsv, head, transformer, tokenizer, get_num = 1):
    '''
    Get embeddings from csv for a particular label
    '''
    headcsv = mycsv['Variable'] == head
    headcsv = headcsv.sample(n=get_num, replace=(get_num > mycsv.shape[0]), random_state=1)
    descs = headcsv['Text'].values
    toks = tokenizer.encode(list(descs)).to(device)
    embs = torch.tensor(transformer(toks))
    return embs

def getTextEmbeddings(heads, transformer, tokenizer, use_convirt = False, get_num=1):
    '''
    return dictionary of text embeddings for specified list of labels in heads
    '''
    if use_convirt:
        filename = '/n/data2/hms/dbmi/beamlab/chexpert/convirt-retrieval/text-retrieval/query_custom.csv'
    else:
        filename = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/mimic_label_queries.csv'

    mycsv = pd.read_csv(filename)
    h_embeds = {}
    for h in heads:
        h_embeds[h] = getLabelEmbeddings(mycsv, h, transformer, tokenizer, get_num)
    return h_embeds

class Report_Tokenizer():
    '''
    CXR-BERT tokenization of list of texts
    '''
    def __init__(self, use_cxr_bert = True):
        url = "microsoft/BiomedVLP-CXR-BERT-specialized"
        self.tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
    def encode(self, texts):
        texts = [t for t in texts]
        encodings = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=texts,
                                        add_special_tokens=True,
                                        padding=True, truncation=True, max_length=256,
                                        return_tensors='pt')
        return encodings

class Transformer_Embeddings(nn.Module):
    '''
    Transformer model with given embedding size, outputting projected CLS embedding
    '''
    def __init__(self, embed_dim = 128):
        super().__init__()
        url = "microsoft/BiomedVLP-CXR-BERT-specialized"
        self.model = AutoModel.from_pretrained(url, trust_remote_code=True)
        self.linear1 = nn.Linear(128, embed_dim)
        modules = [self.model.bert.embeddings, *self.model.bert.encoder.layer[:8]]
        #modules = [self.model.bert.embeddings, *self.model.bert.encoder.layer[:], self.model.cls, self.model.cls_projection_head]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
    def forward(self, text):
        embeddings = self.model.get_projected_text_embeddings(input_ids=text.input_ids,
                                                 attention_mask=text.attention_mask)
        if self.embed_dim != 128:
            embeddings = self.linear1(embeddings)
        return embeddings