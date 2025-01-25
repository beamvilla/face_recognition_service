import os
import numpy as np
from torchvision import transforms
import torch
from typing import List
from PIL import Image


def get_sample_per_person(dataset_path: str) -> int:
    # Get sample per person with min face numbers
    sample_per_person = None
    for face_folder in os.listdir(dataset_path):
        face_dir = os.path.join(dataset_path, face_folder)
        n_face_sample = len(os.listdir(face_dir))

        if n_face_sample == 0:
            raise ValueError(f"The dataset includes empty face folder name {face_folder}")

        if sample_per_person is None:
            sample_per_person = n_face_sample
            continue

        if n_face_sample < sample_per_person:
            sample_per_person = n_face_sample
    
    return sample_per_person


def transform_image(
    dataset_path: str,
    transform_image_size: List[int] = [224, 224]
) -> torch.Tensor:
    image_transform = transforms.Compose([
        transforms.Resize(transform_image_size),
        transforms.ToTensor(),
    ])

    face_folders = os.listdir(dataset_path)
    n_faces = len(face_folders)
    sample_per_person = get_sample_per_person(dataset_path)
    image_tensor = torch.empty(
        n_faces, 
        sample_per_person, 
        3,
        transform_image_size[0], 
        transform_image_size[1]
    )

    for i, face_folder in enumerate(face_folders):
        face_dir = os.path.join(dataset_path, face_folder)

        for j, image_filename in enumerate(os.listdir(face_dir)):
            image_path = os.path.join(face_dir, image_filename)
            image = Image.open(image_path).convert("RGB")
            image_tensor[i, j] = image_transform(image)
    return image_tensor


image_tensor = transform_image(dataset_path="./dataset/face/train")
print(type(image_tensor))
print(image_tensor)

