import torch
import os
from PIL import Image

from models import (
    TripletNet, 
    Resnet34FeatureExtractor, 
    transform_image, 
    ImageMode
)

from utils import save_json


def get_image_feature(model: TripletNet, image: Image) -> torch.Tensor:
    image_tensor = transform_image(
        image=image,
        transform_image_size=[224, 224],
        image_mode=ImageMode.RGB.name
    )

    test_image = torch.stack([image_tensor[0, 0]])
    image_feature = model.feature_extractor(test_image)
    return image_feature


if __name__ == "__main__":
    device = torch.device("cpu")
    model = TripletNet(
        feature_extractor_module=Resnet34FeatureExtractor(n_chanel=3, feat_dim=128, weights=None),
        device=device
    )

    model.load_state_dict(torch.load("./models/model.pt"))

    model.eval()

    images_dir = "./dataset/face/test"

    store_face_features = {}
    for face_name in os.listdir(images_dir):
        face_dir = os.path.join(images_dir, face_name)
        store_face_features[face_name] = []

        for image_filename in os.listdir(face_dir):
            image_path = os.path.join(face_dir, image_filename)
            image = Image.open(image_path)
            image_feature = get_image_feature(model=model, image=image)
            store_face_features[face_name].append(image_feature)
    
    torch.save(store_face_features, "./data/mockdb.pth")