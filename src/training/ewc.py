import torch

class EWC:
    """
    Elastic Weight Consolidation (EWC) implementation.
    """
    def __init__(self, model, ewc_lambda=1000):
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
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            self.model.zero_grad()
            embeddings = self.model(images, domain)
            
            loss, _, _ = self.model.compute_loss_and_acc(embeddings, labels, n_way, k_shot, q_query)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_info[n] += p.grad.data ** 2
            
            num_episodes += 1
            if num_episodes >= 50: 
                break
                
        fisher_info = {n: f / num_episodes for n, f in fisher_info.items()}
        return fisher_info
        
    def remember_task(self, task_name, dataloader, device, n_way=5, k_shot=1, q_query=15):
        """
        Called after training a task. Stores optimal weights and FIM.
        """
        fim = self.compute_fisher_information(dataloader, device, task_name, n_way, k_shot, q_query)
        self.fisher_information[task_name] = fim
        self.params[task_name] = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        
    def penalty(self, model):
        """
        Computes the EWC loss penalty.
        """
        loss = 0.0
        for task_name in self.params:
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.params[task_name]:
                    fisher = self.fisher_information[task_name][name]
                    old_param = self.params[task_name][name]
                    loss += (fisher * (param - old_param) ** 2).sum()
        
        return self.ewc_lambda / 2 * loss
