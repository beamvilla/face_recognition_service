import torch
from PIL import Image

from models import (
    TripletNet, 
    Resnet34FeatureExtractor, 
    transform_image, 
    ImageMode,
    get_features_distance
)


def get_image_feature(model: TripletNet, image: Image) -> torch.Tensor:
    image_tensor = transform_image(
        image=image,
        transform_image_size=[224, 224],
        image_mode=ImageMode.RGB.name
    )

    test_image = torch.stack([image_tensor[0, 0]])
    image_feature = model.feature_extractor(test_image)
    return image_feature


device = torch.device("cpu")
model = TripletNet(
    feature_extractor_module=Resnet34FeatureExtractor(n_chanel=3, feat_dim=128, weights=None),
    device=device
)

model.load_state_dict(torch.load("./models/model.pt"))

model.eval()

image = Image.open("./dataset/face/test/kanghanna/download.jpg")
image_feature = get_image_feature(model=model, image=image)
mockdb = torch.load("./data/mockdb.pth")

min_distance = None
min_distance_face_name = None

for name, stored_face_features in mockdb.items():
    for _feature in stored_face_features:
        distance = get_features_distance(image_feature, _feature).item()
        
        if min_distance is None or distance < min_distance:
            min_distance = distance
            min_distance_face_name = name

print(min_distance_face_name)