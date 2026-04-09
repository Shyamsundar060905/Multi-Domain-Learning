import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network wrapper over a backbone embedding model.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, domain):
        # We assume the backbone outputs normalized embeddings
        embeddings = self.backbone(x, domain)
        return embeddings

    def compute_loss_and_acc(self, embeddings, labels, n_way, k_shot, q_query):
        """
        Computes the prototypical loss and accuracy for an episode.
        Assumes embeddings are structured as:
         [support_class1... support_classN, query_class1... query_classN]
        where support has length k_shot and query has length q_query.
        """
        # Split embeddings into support and query
        num_support = n_way * k_shot
        support_embs = embeddings[:num_support]
        query_embs = embeddings[num_support:]
        
        query_labels = labels[num_support:]

        # Reshape support embeddings to (n_way, k_shot, embedding_dim)
        support_embs = support_embs.view(n_way, k_shot, -1)
        
        # Calculate prototypes (n_way, embedding_dim)
        prototypes = support_embs.mean(dim=1)
        
        # Calculate euclidean distances between queries and prototypes
        # query_embs: (n_way * q_query, emb_dim)
        # prototypes: (n_way, emb_dim)
        dists = torch.cdist(query_embs, prototypes) # shape: (n_query, n_way)
        
        # The true log-probabilities are the negative distances
        log_p_y = F.log_softmax(-dists, dim=1)
        
        # The target indices for queries matching prototypes
        target_inds = torch.arange(n_way).repeat_interleave(q_query).to(embeddings.device)
        
        loss = F.nll_loss(log_p_y, target_inds)
        
        _, y_hat = log_p_y.max(1)
        acc = (y_hat == target_inds).float().mean()
        
        return loss, acc
