import torch

class EWC:
    """
    Elastic Weight Consolidation (EWC) implementation.
    """
    def __init__(self, model, ewc_lambda=1e4):
        self.model = model
        self.ewc_lambda = ewc_lambda
        # Dictionary to store the optimal parameters of old tasks
        self.params = {}
        # Dictionary to store the Fisher Information Matrix (FIM)
        self.fisher_information = {}
        
    def compute_fisher_information(self, dataloader, device, domain, n_way=5, k_shot=1, q_query=15):
        """
        Computes the fisher information matrix for the current task.
        """
        self.model.eval()
        fisher_info = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        num_episodes = 0
        
        for batch in dataloader:
            (s_img1, s_img2, s_mask,
             q_img1, q_img2, q_mask) = batch
        
            q_img1 = q_img1.to(device)
            q_img2 = q_img2.to(device)
            q_mask = q_mask.to(device).float()
        
            self.model.zero_grad()
        
            logits = self.model(q_img1, q_img2, domain)
        
            pos_weight = (q_mask == 0).float().sum() / (q_mask == 1).float().sum().clamp(min=1)

            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, q_mask, pos_weight=pos_weight
            )
        
            loss.backward()
        
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_info[n] += p.grad.data ** 2
        
            num_episodes += 1
            if num_episodes >= 50:
                break        
        
        fisher_info = {n: f for n, f in fisher_info.items()}
        return fisher_info
        
    def remember_task(self, task_name, dataloader, device, n_way=5, k_shot=1, q_query=15):
        """
        Called after training a task. Stores optimal weights and FIM.
        """
        fim = self.compute_fisher_information(dataloader, device, task_name, n_way, k_shot, q_query)
        self.fisher_information[task_name] = fim
        self.params[task_name] = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
        
    def penalty(self, model):
        """
        Computes the EWC loss penalty.
        """
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for task_name in self.params:
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.params[task_name]:
                    fisher = self.fisher_information[task_name][name]
                    old_param = self.params[task_name][name]
                    loss += (fisher * (param - old_param) ** 2).mean()
        
        return self.ewc_lambda / 2 * loss
