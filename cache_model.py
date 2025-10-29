from transformers import AutoModel
MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m" 

if __name__ == "__main__":
    AutoModel.from_pretrained(MODEL_NAME)