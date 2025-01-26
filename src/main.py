import torch
from PIL import Image

from utils import (
    TripletNet, 
    Resnet34FeatureExtractor, 
    transform_image, 
    ImageMode
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

image = Image.open("./dataset/face/test/suzy/images.jpg")
image_feature = get_image_feature(model=model, image=image)