import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

class SiglipEmbedder:
    def __init__(self, model_name="google/siglip-base-patch16-512", device="cpu"):
        print(f"Loading {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        
    def encode(self, text):
        """Encodes text into a normalized SigLIP vector."""
        if isinstance(text, str):
            text = [text]
            
        inputs = self.processor(text=text, padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # SigLIP requires manual L2 normalization for cosine similarity
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
        if len(text) == 1:
            return text_features[0].cpu().numpy()
        return text_features.cpu().numpy()

    def encode_image(self, image):
        """Helper function if you need to re-index your images into Qdrant."""
        # If a file path is provided as a string, open it with PIL
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
        return image_features[0].cpu().numpy()
