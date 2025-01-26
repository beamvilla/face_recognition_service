import os
import sys
sys.path.append("./")

from typing import List
import torch
import torch.optim as optim
from tqdm import tqdm

from utils import get_logger
from models import (
    transform_images_batch, 
    TripletImageLoader, 
    TripletNet, 
    Resnet34FeatureExtractor,
    TripletLoss,
)
from config.train_config import TrainConfig


def train(
    train_dir: str,
    test_dir: str,
    model_dir: str, 
    n_test_samples: int,
    n_val: int,
    epochs: int = 100, 
    batch_size: int = 4, 
    eval_every: int = 20, 
    loss_every: int = 20,
    loss_alpha: int = 0.8,
    learning_rate: float = 2e-5,
    input_size: List[int] = [224, 224]
):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "model.pt")

    image_tensors_train = transform_images_batch(
        dataset_path=train_dir,
        transform_image_size=input_size
    )
    image_tensors_test = transform_images_batch(
        dataset_path=test_dir,
        transform_image_size=input_size
    )

    triplet_image_loader = TripletImageLoader(image_tensors_train, image_tensors_test)
    model = TripletNet(
        feature_extractor_module=Resnet34FeatureExtractor(n_chanel=3, feat_dim=128, weights=None),
        device=device
    )
    loss_func = TripletLoss(loss_alpha)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for i in tqdm(range(epochs)):
        xb = triplet_image_loader.get_batch(batch_size)
        loss = loss_func(model(xb))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        best_acc = 0

        # evaluate
        if (i % eval_every == 0) and (i != 0):
            val_acc = triplet_image_loader.test_oneshot(model, n_test_samples, n_val)
            get_logger().info(f"validation accuracy on {n_test_samples} supports of total {n_val} set:{val_acc}")
            if val_acc >= best_acc:
                get_logger().info("saving")
                torch.save(model.state_dict(), model_path)
                best_acc = val_acc

        if i % loss_every == 0:
            get_logger().info("iteration {}, training loss: {:.2f},".format(i, loss.item()))
 

if __name__ == "__main__":
    train_config = TrainConfig("./config/train.yaml")

    device = torch.device(train_config.DEVICE)
    get_logger().info(f"Model training on {device}")
   
    n_train_faces = len(os.listdir(train_config.TRAIN_DIR))
    n_val_faces = len(os.listdir(train_config.TEST_DIR))

    train(
        train_dir=train_config.TRAIN_DIR,
        test_dir=train_config.TEST_DIR,
        model_dir=train_config.MODEL_DIR,
        n_test_samples=n_val_faces,
        n_val=n_val_faces,
        epochs=train_config.EPOCHS,
        batch_size=n_train_faces, 
        eval_every=train_config.EPOCHS // 2, 
        loss_every=train_config.EPOCHS // 2,
        loss_alpha=train_config.LOSS_ALPHA,
        learning_rate=train_config.LEARNING_RATE,
        input_size=[train_config.INPUT_WIDTH, train_config.INPUT_HEIGHT]
    )