import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils.helpers import freeze_domain
from src.training.ewc import EWC

class ContinualFewShotTrainer:
    def __init__(self, model, train_loaders, test_loaders, domain_list, optimizers, schedulers, device, ewc_lambda=1000):
        self.model = model
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.domain_list = domain_list
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.device = device
        self.ewc = EWC(model, ewc_lambda=ewc_lambda)

    def train_task(self, domain, epochs, n_way, k_shot, q_query):
        """Trains a single domain sequentially, then stores it in EWC."""
        if domain not in self.train_loaders:
            return
            
        loader = self.train_loaders[domain]
        print(f"\n--- Training on domain: {domain} ---")
        
        for epoch in range(epochs):
            self.model.train()
            domain_loss = 0.0
            total_acc = 0.0
            num_episodes = 0

            # Only unfreeze current domain adapter; rest is managed automatically
            freeze_domain(self.model, domain)
            
            progress_bar = tqdm(loader, desc=f"[{domain}] Epoch {epoch+1}/{epochs}", leave=True)

            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizers[domain].zero_grad()
                
                # Forward pass - returns normalized embeddings
                embeddings = self.model(images, domain)
                
                loss, acc = self.model.compute_loss_and_acc(embeddings, labels, n_way, k_shot, q_query)
                
                # EWC Penalty
                ewc_loss = self.ewc.penalty(self.model)
                total_loss_batch = loss + ewc_loss
                
                total_loss_batch.backward()
                self.optimizers[domain].step()

                domain_loss += loss.item()
                total_acc += acc.item()
                num_episodes += 1

                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ewc': f'{ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else ewc_loss:.4f}',
                    'acc': f'{acc.item() * 100:.2f}%' 
                })

            avg_domain_loss = domain_loss / max(num_episodes, 1)
            avg_domain_acc = total_acc / max(num_episodes, 1) * 100
            print(f"Domain: {domain}, Epoch Loss: {avg_domain_loss:.4f}, Accuracy: {avg_domain_acc:.2f}%")

            self.schedulers[domain].step()
            self.evaluate(domain, n_way, k_shot, q_query)
        
        # After training, tell EWC to remember
        print(f"Consolidating weights for {domain}...")
        self.ewc.remember_task(domain, loader, self.device, n_way, k_shot, q_query)

    def evaluate(self, domain, n_way, k_shot, q_query):
        self.model.eval()
        
        if domain not in self.test_loaders:
            return 0.0
            
        with torch.no_grad():
            total_acc = 0.0
            num_episodes = 0
            
            print(f"Evaluating {domain}...")
            loader = self.test_loaders[domain]
            progress_bar = tqdm(loader, desc=f"{domain} Eval", leave=False)

            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                embeddings = self.model(images, domain)

                _, acc = self.model.compute_loss_and_acc(embeddings, labels, n_way, k_shot, q_query)
                total_acc += acc.item()
                num_episodes += 1

                progress_bar.set_postfix(acc=f"{acc.item() * 100:.2f}%")

            accuracy = total_acc / max(num_episodes, 1) * 100
            print(f"Final Accuracy on {domain}: {accuracy:.2f}%")
            return accuracy
            
    def evaluate_all(self, n_way, k_shot, q_query):
        """Evaluate on all seen tasks to measure forgetting."""
        print("\n--- Evaluating Forgetting Across All Domains ---")
        results = {}
        for domain in self.domain_list:
            if domain in self.test_loaders:
                acc = self.evaluate(domain, n_way, k_shot, q_query)
                results[domain] = acc
        return results
