import torch
from torch import nn

class CNN_Embeddings(nn.Module): #image -> embed size
    def __init__(self, embed_dim):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.resnet.fc = nn.Linear(2048, embed_dim)

    def forward(self, image):
        im_embeddings = self.resnet(image)
        return im_embeddings

class CNN_Classifier(nn.Module):
    def __init__(self, embed_dim,cnn_model=None, freeze=True, num_heads=5):
        super().__init__()
        freeze = freeze and cnn_model #don't freeze a blank model
        for param in cnn_model.parameters():
            param.requires_grad = (not freeze) #freeze the parameters if freeze is true

        self.cnn_model = cnn_model if cnn_model else CNN_Embeddings(embed_dim)
        self.relu = nn.ReLU()
        self.classification_head = nn.Linear(embed_dim, num_heads)

    def forward(self, image):
        embedding = self.cnn_model(image)
        output = self.relu(embedding)
        output = self.classification_head(output)
        return output


