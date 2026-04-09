import numpy as np
import torch
from torch.utils.data import Sampler

class EpisodicBatchSampler(Sampler):
    """
    Sampler for N-way K-shot Q-query episodic training.
    Yields batches of indices where each batch contains:
    n_way classes, structured as:
    support samples (k_shot per class) followed by query samples (q_query per class).
    Total batch size = n_way * (k_shot + q_query)
    """
    def __init__(self, labels, n_way, k_shot, q_query, num_episodes):
        super().__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes
        
        # Determine unique classes and indices for each class
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        elif isinstance(labels, list):
            labels = np.array(labels)
            
        self.classes = np.unique(labels)
        
        # In a generic multi-domain framework, if some tasks don't have enough classes
        # for n_way, adjust n_way dynamically or assert.
        if len(self.classes) < self.n_way:
            print(f"[Warning] Only {len(self.classes)} classes available, modifying n_way from {self.n_way} to {len(self.classes)}")
            self.n_way = len(self.classes)

        self.indices_per_class = {c: np.where(labels == c)[0] for c in self.classes}
        
    def __len__(self):
        return self.num_episodes
        
    def __iter__(self):
        for _ in range(self.num_episodes):
            batch_support = []
            batch_query = []
            
            # Select n_way classes randomly
            classes_in_episode = np.random.choice(self.classes, self.n_way, replace=False)
            
            for c in classes_in_episode:
                indices = self.indices_per_class[c]
                num_needed = self.k_shot + self.q_query
                
                # Sample
                if len(indices) < num_needed:
                    selected = np.random.choice(indices, num_needed, replace=True)
                else:
                    selected = np.random.choice(indices, num_needed, replace=False)
                
                batch_support.extend([int(x) for x in selected[:self.k_shot]])
                batch_query.extend([int(x) for x in selected[self.k_shot:]])
            
            # Yield support indices followed by query indices.
            yield batch_support + batch_query
