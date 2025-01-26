from src.models import TripletNet, ImageMode, transform_image, Resnet34FeatureExtractor
from utils import get_logger
from PIL import Image
import io
import torch
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler


class TripletNetHandler(BaseHandler):
    def initialize(self, context):
        # Load the model
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model = TripletNet(
            Resnet34FeatureExtractor(n_chanel=3, feat_dim=128, weights=None), 
            self.device
        )
        self.model.load_state_dict(torch.load("./models/model.pt"))
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        # Preprocess input data
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        images = [transform(Image.open(io.BytesIO(item["data"]))) for item in data]
        return torch.stack(images).to(self.device)

    def inference(self, inputs):
        # Extract features
        with torch.no_grad():
            features = self.model.feature_extractor(inputs)
        return features.cpu().numpy().tolist()

    def postprocess(self, inference_results):
        # Return results as JSON
        return [{"features": result} for result in inference_results]