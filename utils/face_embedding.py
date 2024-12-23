from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torch


mtcnn = MTCNN()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# facenet_model = InceptionResnetV1(device=device)
# state_dict = torch.load("data2/facenet_triplet_finetuned.pth", map_location=device)
# filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("logits")}
# facenet_model.load_state_dict(filtered_state_dict)
# facenet_model.eval()

facenet_model = InceptionResnetV1(pretrained="vggface2").eval()

async def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    face = mtcnn(image)
    return face

async def get_facenet_embedding(face):
    if face is None:
        raise ValueError(f"На фото нет лица")
    with torch.no_grad():
        embedding = facenet_model(face.unsqueeze(0))
    return embedding.squeeze().numpy()

MODELS = {
    "facenet": get_facenet_embedding
}

async def get_face_embedding(face, model_name = "facenet"):
    if model_name not in MODELS:
        raise ValueError(f"Модель '{model_name}' не поддерживается. Поддерживаемые модели: {list(MODELS.keys())}")
    return await MODELS[model_name](face)