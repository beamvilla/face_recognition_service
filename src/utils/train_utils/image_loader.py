import numpy as np
import torch
from typing import List, Tuple

from utils.train_utils.model import TripletNet


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

    def make_oneshot_task(self, n_test_samples: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        random_face_indexs = np.random.choice(self.n_val, size=(n_test_samples,), replace=False)
        indices = np.random.randint(0, self.n_samples, size=(n_test_samples,))
        true_face = random_face_indexs[0]

        ex1, ex2 = np.random.choice(self.n_samples, replace=False, size=(2,))
        test_image = torch.stack([self.image_tensors_val[true_face, ex1]] * n_test_samples)
        support_set = self.image_tensors_val[random_face_indexs, indices]
        support_set[0] = self.image_tensors_val[true_face, ex2]
        pairs = [test_image, support_set]

        targets = torch.zeros((n_test_samples,))
        targets[0] = 1
        return pairs, targets
    
    def test_oneshot(self, model: TripletNet, n_test_samples: int, k: int) -> float:
        model.eval()
        n_correct = 0
        for _ in range(k):
            inputs, _ = self.make_oneshot_task(n_test_samples)
            dists = model.get_distance(*inputs).cpu().detach().numpy()
            if np.argmin(dists) == 0:
                n_correct += 1
        pct_correct = (100 * n_correct / k)
        return pct_correct