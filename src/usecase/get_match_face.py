import torch
from PIL import Image
from typing import List

from src.models import (
    Resnet34FeatureExtractor, 
    TripletNet, 
    ImageMode, 
    transform_image,
    get_features_distance
)


class GetMatchFace:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model = TripletNet(
            feature_extractor_module=Resnet34FeatureExtractor(n_chanel=3, feat_dim=128, weights=None),
            device=self.device
        )

        self.model.load_state_dict(torch.load("./models/model.pt"))
        self.model.eval()

        self.mockdb = torch.load("./data/mockdb.pth")
    
    def get_image_feature(
        self, 
        image: Image,
        transform_image_size: List[int] = [224, 224]
    ) -> torch.Tensor:
        image_tensor = transform_image(
            image=image,
            transform_image_size=transform_image_size,
            image_mode=ImageMode.RGB.name
        )

        test_image = torch.stack([image_tensor[0, 0]])
        image_feature = self.model.feature_extractor(test_image)
        return image_feature
    
    def get_match_face(self, image_feature: torch.Tensor) -> str:
        min_distance = None
        match_face = None

        for name, stored_face_features in self.mockdb.items():
            for _feature in stored_face_features:
                distance = get_features_distance(image_feature, _feature).item()
                
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    match_face = name

        return match_face
    
    def predict(self, image: Image) -> str:
        image_feature = self.get_image_feature(image=image)
        match_face = self.get_match_face(image_feature)
        return match_face