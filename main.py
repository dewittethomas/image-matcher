import torch
from PIL import Image
import requests

from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding

img1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
img2 = "https://imgs.search.brave.com/ENBYGHUAFfpfS1iQTDZGy0grc3uFNtv8wTCT64rhtak/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5nZXR0eWltYWdl/cy5jb20vaWQvMjIx/NTkxMjI5My9waG90/by9hLWdyZWF0LWRh/bmUtZG9nLXNpdHMt/d2l0aC1jaGlodWFo/dWEtZG9ncy1vbi1h/LXNvZmEtb2YtdGhl/LWRvZ3MtYW5kLWZ1/bi1mYWlyLWF0LXRo/ZS5qcGc_cz02MTJ4/NjEyJnc9MCZrPTIw/JmM9SXExSC1JT2VT/VGd5ZnpIMHA1MHpK/Qi1LdGlJN1Q2YXhI/QndiY3gxVzEzYz0"

emb1 = get_embedding(requests.get(img1, stream=True).raw)
emb2 = get_embedding(requests.get(img2, stream=True).raw)

similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
print(f"Cosine similarity: {similarity.item():.4f}")