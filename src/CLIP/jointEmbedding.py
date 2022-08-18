import CNN
import Transformer
from HelperFunctions import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class JointEmbeddingModel(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cnn = CNN.CNN_Embeddings(embed_dim=embed_dim)
        self.transformer = Transformer.Transformer_Embeddings(embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image, text):
        image_features = self.cnn(image)
        text_features = self.transformer(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

class CNN_Similarity_Classifier(nn.Module):
    def __init__(self, je_model, embed_size=128, freeze=True,
                 heads=np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']),
                 use_convirt=False, get_num=20,  soft=False):

        super().__init__()
        self.cnn_model = je_model.cnn
        self.transformer_model = je_model.transformer
        for param in self.cnn_model.parameters():
            param.requires_grad = not freeze
        for param in self.transformer_model.parameters():
            param.requires_grad = not freeze

        self.tokenizer = Transformer.Report_Tokenizer()
        self.embed_size = embed_size
        self.heads = heads
        self.device = device
        self.tembed, self.tlab = Transformer.getTextEmbeddings(heads=self.heads, transformer=self.transformer_model, tokenizer=self.tokenizer,
                                                               use_convirt=use_convirt, device=device,
                                                               get_num=get_num)
        self.get_num = get_num
        self.softmax = nn.Softmax(dim=1)
        self.soft = soft
    def forward(self, image):
        im_embedding = self.cnn_model(image)
        im_embedding = im_embedding / im_embedding.norm(dim=-1, keepdim=True)
        class_score = torch.zeros(im_embedding.shape[0], self.heads.shape[0]).to(self.device)
        for i, h in enumerate(self.heads):
            t = self.tembed[h]
            tembed = t / t.norm(dim=-1, keepdim=True)
            if self.get_num > 1:
                tembed = tembed.mean(dim=0)
            tembed = tembed/tembed.norm(dim=-1, keepdim=True)
            head_sim = im_embedding @ tembed.t()
            head_sim = head_sim.squeeze()
            class_score[:, i] = head_sim
        if self.soft:
            return self.softmax(class_score)
        else:
            return class_score

if __name__ == '__main__':
    je_model = JointEmbeddingModel(embed_dim=128).to(device)
    mimic_dat = getDatasets(subset=['tinytrain'], augs=2)
    [train_data_loader_mimic] = getLoaders(mimic_dat, subset=['tinytrain'], num_work=16)
    params = list(je_model.parameters())
    optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.000001)
    tokenizer = Transformer.Report_Tokenizer()
    je_model.train()
    for i, res in enumerate(train_data_loader_mimic):
        loss = train(je_model, res)
        loss.backward()
        optimizer.step()
        break