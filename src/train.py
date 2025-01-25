import os
import torch
from utils import (
    transform_images, 
    TripletImageLoader, 
    TripletNet, 
    Resnet34FeatureExtractor,
    TripletLoss
)
import torch.optim as optim
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    learning_rate: float = 2e-5
):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "model.pt")

    image_tensors_train = transform_images(dataset_path=train_dir)
    image_tensors_test = transform_images(dataset_path=test_dir)

    triplet_image_loader = TripletImageLoader(image_tensors_train, image_tensors_test)
    model = TripletNet(
        feature_extractor_module=Resnet34FeatureExtractor(n_chanel=3, feat_dim=128, pretrained=False),
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
            print(f"validation accuracy on {n_test_samples} supports of total {n_val} set:{val_acc}")
            if val_acc >= best_acc:
                print("saving")
                torch.save(model.state_dict(), model_path)
                best_acc = val_acc

        if i % loss_every == 0:
            print("iteration {}, training loss: {:.2f},".format(i, loss.item()))
 

train_dir = "./dataset/face/train"
test_dir = "./dataset/face/test"

n_val_faces = len(os.listdir(test_dir))
n_train_faces = len(os.listdir(train_dir))
epochs = 10

train(
    train_dir=train_dir,
    test_dir=test_dir,
    model_dir="./models/",
    n_test_samples=n_val_faces,
    n_val=n_val_faces,
    epochs=epochs,
    batch_size=n_train_faces, 
    eval_every=epochs // 2, 
    loss_every=epochs // 2,
    loss_alpha=0.8,
    learning_rate=2e-5
)