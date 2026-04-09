import os
import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils.helpers import freeze_domain
from src.training.ewc import EWC

class ContinualFewShotTrainer:
    def __init__(self, model, train_loaders, test_loaders, domain_list, optimizers, schedulers, device, ewc_lambda=1000, output_dir='./checkpoints'):
        self.model = model
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.domain_list = domain_list
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.device = device
        self.ewc = EWC(model, ewc_lambda=ewc_lambda)
        self.output_dir = output_dir
        self.amp_device_type = self.device.type
        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.amp_device_type, enabled=self.use_amp)

    def train_task(self, domain, epochs, n_way, k_shot, q_query):
        """Trains a single domain sequentially, then stores it in EWC."""
        if domain not in self.train_loaders:
            return
            
        loader = self.train_loaders[domain]
        print(f"\n--- Training on domain: {domain} ---")
        
        best_acc = 0.0
        for epoch in range(epochs):
            self.model.train()
            domain_loss = 0.0
            total_acc = 0.0
            total_f1 = 0.0
            num_episodes = 0

            # Only unfreeze current domain adapter; rest is managed automatically
            freeze_domain(self.model, domain)
            
            progress_bar = tqdm(loader, desc=f"[{domain}] Epoch {epoch+1}/{epochs}", leave=True)

            for images, labels in progress_bar:
                try:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    self.optimizers[domain].zero_grad(set_to_none=True)

                    with torch.amp.autocast(device_type=self.amp_device_type, enabled=self.use_amp):
                        embeddings = self.model(images, domain)
                        loss, acc, f1 = self.model.compute_loss_and_acc(embeddings, labels, n_way, k_shot, q_query)
                        ewc_loss = self.ewc.penalty(self.model)
                        total_loss_batch = loss + ewc_loss

                    self.scaler.scale(total_loss_batch).backward()
                    self.scaler.step(self.optimizers[domain])
                    self.scaler.update()

                    domain_loss += loss.item()
                    total_acc += acc.item()
                    total_f1 += f1
                    num_episodes += 1

                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'ewc': f'{ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else ewc_loss:.4f}',
                        'acc': f'{acc.item() * 100:.2f}%',
                        'f1': f'{f1:.3f}'
                    })
                except RuntimeError as err:
                    if "out of memory" in str(err).lower() and self.device.type == "cuda":
                        torch.cuda.empty_cache()
                        print(f"[OOM] Skipping one episode in {domain}: {err}")
                        continue
                    raise

            avg_domain_loss = domain_loss / max(num_episodes, 1)
            avg_domain_acc = total_acc / max(num_episodes, 1) * 100
            avg_domain_f1 = total_f1 / max(num_episodes, 1)
            print(
                f"Domain: {domain}, Epoch: {epoch + 1}/{epochs}, "
                f"Loss: {avg_domain_loss:.4f}, Accuracy: {avg_domain_acc:.2f}%, F1: {avg_domain_f1:.3f}"
            )

            self.schedulers[domain].step()
            val_acc = self.evaluate(domain, n_way, k_shot, q_query)
            
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(self.output_dir, f'best_model_{domain}.pth')
                torch.save(self.model.state_dict(), save_path)
                print(f" -> Saved new best checkpoint for {domain} (Acc: {val_acc:.2f}%)")
        
        # After training, tell EWC to remember
        print(f"Consolidating weights for {domain}...")
        self.ewc.remember_task(domain, loader, self.device, n_way, k_shot, q_query)

    def evaluate(self, domain, n_way, k_shot, q_query):
        self.model.eval()
        
        if domain not in self.test_loaders:
            return 0.0
            
        with torch.no_grad():
            total_acc = 0.0
            total_f1 = 0.0
            num_episodes = 0
            
            print(f"Evaluating {domain}...")
            loader = self.test_loaders[domain]
            progress_bar = tqdm(loader, desc=f"{domain} Eval", leave=False)

            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                embeddings = self.model(images, domain)

                _, acc, f1 = self.model.compute_loss_and_acc(embeddings, labels, n_way, k_shot, q_query)
                total_acc += acc.item()
                total_f1 += f1
                num_episodes += 1

                progress_bar.set_postfix(acc=f"{acc.item() * 100:.2f}%", f1=f"{f1:.3f}")

            accuracy = total_acc / max(num_episodes, 1) * 100
            f1_score = total_f1 / max(num_episodes, 1)
            print(f"Final metrics on {domain}: accuracy={accuracy:.2f}%, f1={f1_score:.3f}")
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
