from torch import nn
from transformers import AutoModel, AutoConfig
import os
import torch
from datetime import datetime
from load_weights_from_r2 import load_weights_from_r2

class HotModel(nn.Module):
    def __init__(self, backbone='facebook/dinov3-vit7b16-pretrain-lvd1689m', bb_dim=4096) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        # self.backbone = AutoModel.from_config(AutoConfig.from_pretrained(backbone))
        self.proj = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(bb_dim, 2 * bb_dim),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.Linear(2 * bb_dim, 1)
        )
        self.load_finetune_weights('./weights')
        
    def forward(self, x):
        x = self.backbone(x).pooler_output
        x = self.proj(x).squeeze(1)
        return x
    
    def load_finetune_weights(self, folder_name):
        if not os.path.exists(folder_name):
            print(f"Folder '{folder_name}' not found. Downloading weights...")
            load_weights_from_r2(local_dir=folder_name)
        
        proj_path = os.path.join(folder_name, "proj.pt")
        layer_path = os.path.join(folder_name, "layer_-1.pt")
        norm_path = os.path.join(folder_name, "norm.pt")
        
        assert os.path.exists(proj_path), f"{proj_path} doesn't exist"
        assert os.path.exists(layer_path), f"{layer_path} doesn't exist"
        assert os.path.exists(norm_path), f"{norm_path} doesn't exist"
        self.proj.load_state_dict(torch.load(proj_path, map_location='cpu'))
        self.backbone.layer[-1].load_state_dict(torch.load(layer_path, map_location='cpu'))
        self.backbone.norm.load_state_dict(torch.load(norm_path, map_location='cpu'))

    def save_finetune_weights(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"ft_weights_{timestamp}"
        os.makedirs(folder_name, exist_ok=True)
        torch.save(self.proj.state_dict(), os.path.join(folder_name, "proj.pt"))
        torch.save(self.backbone.layer[-1].state_dict(), os.path.join(folder_name, "layer_-1.pt"))
        torch.save(self.backbone.norm.state_dict(), os.path.join(folder_name, "norm.pt"))