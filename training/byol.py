
import random
import copy
from PIL import ImageFilter

import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torch.nn import functional as F



class NormalizedMSELoss(nn.Module):
    def __init__(self) -> None:
        super(NormalizedMSELoss,self).__init__()

    def forward(self, view1, view2):
        v1 = F.normalize(view1, dim=-1)
        v2 = F.normalize(view2, dim=-1)
        return 2 - 2 * (v1 * v2).sum(dim=-1)
    
    
class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img
        
        
class BYOL_Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomApply(
                [transforms.RandomResizedCrop(224, scale=(0.2, 0.9))],
                p=0.8
            ),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.8, contrast=0.8)],
                p=0.2
            ),
            transforms.RandomRotation(30),
            GaussianBlur(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_prime = transforms.Compose([
            transforms.RandomApply(
                [transforms.RandomResizedCrop(224, scale=(0.6, 0.9))],
                p=0.2
            ),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4)],
                p=0.8
            ),
            transforms.RandomRotation(30),
            GaussianBlur(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2
    

def get_encoder_model(**kwargs):
    resnet = torchvision.models.resnet50(**kwargs)
    resnet.fc = torch.nn.Identity() 
    return resnet
    

class MLP(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int=128, hidden_dim: int=512):
        super(MLP,self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        return self.net(x)
    
    
class EncoderProjecter(nn.Module):
    def __init__(self,
                    encoder: nn.Module,
                    hidden_dim: int=512,
                    projection_out_dim: int=128
                    ) -> None:
        super(EncoderProjecter, self).__init__()

        self.encoder = encoder # encoder is ResNet50 with classifier head removed (Linear layer) 
        self.projection = MLP(input_dim=2048, projection_dim=projection_out_dim, hidden_dim=hidden_dim)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1]) # flattens
        return self.projection(h)
    
class BYOL(nn.Module):
    def __init__(self,
                    hidden_dim: int = 512,
                    projection_out_dim: int = 128,
                    target_decay: float = 0.996
                ) -> None:
        super(BYOL, self).__init__()
        resnet = get_encoder_model(weights=ResNet50_Weights.IMAGENET1K_V1)

        # freeze everything
        for param in resnet.parameters():
            param.requires_grad = False

        # unfreeze layer3 + layer4
        for param in resnet.layer3.parameters():
            param.requires_grad = True

        for param in resnet.layer4.parameters():
            param.requires_grad = True
    
    
        self.online_network = EncoderProjecter(resnet)  # encoder + projector
        self.online_predictor = MLP(input_dim=projection_out_dim, projection_dim=projection_out_dim, hidden_dim=hidden_dim)

        self.target_network = copy.deepcopy(self.online_network)  # independent copy
        self.target_network.load_state_dict(self.online_network.state_dict())
        
        # set target_network's weights to be untrainable
        self.target_network.eval()
        for param in self.target_network.parameters():
            param.requires_grad = False
        self.target_decay = target_decay
        self.loss_function = NormalizedMSELoss()


    @torch.no_grad()
    def soft_update_target_network(self) -> None:
        for online_p, target_p in zip(self.online_network.parameters(), self.target_network.parameters()):
            target_p.data = target_p.data * self.target_decay + online_p.data * (1. - self.target_decay)


    def forward(self, view):
        online_proj = self.online_network(view)
        target_proj = self.target_network(view)

        return online_proj, target_proj

    def loss(self, view1, view2):
        online_proj1, target_proj1 = self(view1) # v embedding
        online_proj2, target_proj2 = self(view2) # v' embedding

        online_prediction_1 = self.online_predictor(online_proj1)
        online_prediction_2 = self.online_predictor(online_proj2)

        loss1 = self.loss_function(online_prediction_1, target_proj2.detach()) # online v prediction, target v' prediction
        loss2 = self.loss_function(online_prediction_2, target_proj1.detach()) # online v' prediction, target v prediction
        return torch.mean(loss1 + loss2)
    
    

    
    
class Classifier(nn.Module):
    def __init__(self, encoder_model, frozen=False):
        super(Classifier, self).__init__()
        self.encoder = copy.deepcopy(encoder_model)
        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.fc = nn.Linear(in_features=2048, out_features=2)
        

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc(x)
        return x
