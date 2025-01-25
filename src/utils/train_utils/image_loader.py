import numpy as np
import torch
from typing import List


class TripletImageLoader:
    def __init__(self, image_tensors_train: torch.Tensor, image_tensors_val: torch.Tensor):
        self.image_tensors_train = image_tensors_train
        self.image_tensors_val = image_tensors_val
        self.n_classes, self.n_samples, self.n_chanels, \
            self.height, self.width = image_tensors_train.shape
        self.n_val = image_tensors_val.size(0)
        
    def get_batch(self, n_pick: int) -> List[torch.Tensor]:
        # Get triplet batches include [anchor, positive, negative]

        # Randomly pick face
        random_face_indexs = np.random.choice(self.n_classes, size=(n_pick,), replace=False)
        triplets = [torch.zeros((n_pick, self.n_chanels, self.height, self.width)) for _ in range(3)]

        for i in range(n_pick):
            # Pick anchor class
            face_idx = random_face_indexs[i]

            # Correct anchor and positive face
            idxs = np.random.choice(self.n_samples, size=2, replace=False)
            anchor_idx, pos_idx = idxs[0], idxs[1]
            triplets[0][i] = self.image_tensors_train[face_idx, anchor_idx]
            triplets[1][i] = self.image_tensors_train[face_idx, pos_idx]
            
            # Correct negative face
            face_neg_idx = np.random.choice(list(filter(lambda x: x != face_idx, random_face_indexs)))
            neg_idx = np.random.randint(0, self.n_samples)
            triplets[2][i] = self.image_tensors_train[face_neg_idx, neg_idx]

        return triplets