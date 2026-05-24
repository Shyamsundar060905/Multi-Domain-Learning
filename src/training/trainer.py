import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils.helpers import freeze_domain
from src.training.ewc import EWC
from torch.utils.data import DataLoader


import torch.nn.functional as F


def dice_loss(pred, target):
    smooth = 1e-6
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


    

def few_shot_cd_loss(pred, target):
    pos_pixels = target.sum()
    neg_pixels = target.numel() - pos_pixels
    
    if pos_pixels < 1:
        pos_weight = torch.tensor(1.0, device=target.device)
    else:
        pos_weight = (neg_pixels / pos_pixels + 1e-6) 

        
    bce = F.binary_cross_entropy_with_logits(
        pred, target,
        pos_weight=pos_weight
    )
    dice = dice_loss(pred, target)
    return 0.2 * bce + 0.8 * dice
    

class ContinualFewShotTrainer:
    def __init__(self, model, train_loaders, test_loaders, domain_list, device, ewc_lambda=1e4):
        self.model = model
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.domain_list = domain_list
        self.optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
    )
        self.device = device
        self.ewc = EWC(model, ewc_lambda=ewc_lambda)
        self.amp_device_type = self.device.type
        self.use_amp = False
        self.scaler = torch.amp.GradScaler(self.amp_device_type, enabled=self.use_amp)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
        
    def train_step(self, batch, domain):
        img1, img2, mask = batch
    
        img1 = img1.to(self.device, non_blocking=True)
        img2 = img2.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True).float()

    # Ensure batch dimension exists
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
            mask = mask.unsqueeze(0)

    # 🔥 Freeze domain BEFORE forward
        freeze_domain(self.model, domain)

        self.optimizer.zero_grad()

        with torch.amp.autocast(device_type=self.amp_device_type,enabled=self.use_amp):
            logits = self.model(img1, img2, domain)
            loss = few_shot_cd_loss(logits, mask)
    
            ewc_loss = self.ewc.penalty(self.model)
            total_loss = loss + ewc_loss
    
        # -----------------------------
        # Backprop
        # -----------------------------
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    # -----------------------------
    # Metrics (no grad)
    # -----------------------------
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()
    
            intersection = (pred * mask).sum()
            union = pred.sum() + mask.sum()
            dice = (2 * intersection + 1e-6) / (union + 1e-6)
    
        return loss.item(), ewc_loss.item(), dice.item()

       
    def train_joint(self, epochs, n_way, k_shot, q_query):
        for epoch in range(epochs):
            self.model.train()
    
            loaders = {
                d: iter(self.train_loaders[d])
                for d in self.domain_list
            }
    
            steps_per_epoch = min(len(self.train_loaders[d]) for d in self.domain_list)
    
            progress_bar = tqdm(
                range(steps_per_epoch),
                desc=f"[Joint] Epoch {epoch+1}/{epochs}",
                leave=True
            )
    
            for step in progress_bar:
                for domain in self.domain_list:
                    try:
                        batch = next(loaders[domain])
                    except StopIteration:
                        loaders[domain] = iter(self.train_loaders[domain])
                        batch = next(loaders[domain])
    
                    loss, ewc_loss,dice = self.train_step(batch, domain)
    
                    progress_bar.set_postfix({
                        "domain": domain,
                        "loss": f"{loss:.4f}",
                        "ewc": f"{ewc_loss:.4f}",
                        "dice": f"{dice:.4f}"
                    });
    
        # 🔥 Consolidation
        print("\nConsolidating weights for all domains...")
        for domain in self.domain_list:
            self.ewc.remember_task(
                domain,
                self.train_loaders[domain],
                self.device,
                n_way, k_shot, q_query
            )
    # def train_task(self, domain, epochs, n_way, k_shot, q_query):
    #     """Trains a single domain sequentially for few-shot change detection."""
    
    #     if domain not in self.train_loaders:
    #         return
    
    #     loader = self.train_loaders[domain]
    #     print(f"\n--- Training on domain: {domain} ---")
    
    #     def segmentation_metrics(logits, target):
    #         probs = torch.sigmoid(logits)
    #         pred = (probs > 0.5).float()
        
    #         target = target.float()
        
    #         # Accuracy
    #         acc = (pred == target).float().mean()
        
    #         # Dice / F1
    #         intersection = (pred * target).sum(dim=(1,2,3))
    #         union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    #         dice = (2 * intersection + 1e-6) / (union + 1e-6)
    #         dice = dice.mean()
        
    #         # IoU
    #         iou = (intersection + 1e-6) / (union - intersection + 1e-6)
    #         iou = iou.mean()
        
    #         return acc, dice, iou
    #     for epoch in range(epochs):
    #         self.model.train()
    
    #         domain_loss = 0.0
    #         total_acc = 0.0
    #         total_dice = 0.0
    #         num_episodes = 0
    
    #         freeze_domain(self.model, domain)
    
    #         progress_bar = tqdm(loader, desc=f"[{domain}] Epoch {epoch+1}/{epochs}", leave=True)
    
    #         for batch in progress_bar:
    #             try:
    #                 (s_img1, s_img2, s_mask,
    #                  q_img1, q_img2, q_mask) = batch
    
    #                 s_img1 = s_img1.to(self.device, non_blocking=True)
    #                 s_img2 = s_img2.to(self.device, non_blocking=True)
    #                 s_mask = s_mask.to(self.device, non_blocking=True)
    
    #                 q_img1 = q_img1.to(self.device, non_blocking=True)
    #                 q_img2 = q_img2.to(self.device, non_blocking=True)
    #                 q_mask = q_mask.to(self.device, non_blocking=True).float()
    
    #                 self.optimizers[domain].zero_grad(set_to_none=True)
    
    #                 with torch.amp.autocast(
    #                     device_type=self.amp_device_type,
    #                     enabled=self.use_amp
    #                 ):
    #                     logits = self.model(q_img1, q_img2, domain)
    #                     loss = few_shot_cd_loss(logits, q_mask)
    
    #                     ewc_loss = self.ewc.penalty(self.model)
    #                     total_loss_batch = loss + ewc_loss
    
    #                 self.scaler.scale(total_loss_batch).backward()
    #                 self.scaler.step(self.optimizers[domain])
    #                 self.scaler.update()
    
    #                 acc, dice, iou = segmentation_metrics(logits.detach(), q_mask)
    
    #                 domain_loss += loss.item()
    #                 total_acc += acc.item()
    #                 total_dice += dice.item()
    #                 num_episodes += 1
    
    #                 progress_bar.set_postfix({
    #                     'loss': f'{loss.item():.4f}',
    #                     'ewc': f'{ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else ewc_loss:.4f}',
    #                     'acc': f'{acc.item() * 100:.2f}%',
    #                     'dice': f'{dice.item():.4f}',
    #                     'iou': f'{iou.item():.4f}'

    #                 })
    
    #             except RuntimeError as err:
    #                 if "out of memory" in str(err).lower() and self.device.type == "cuda":
    #                     torch.cuda.empty_cache()
    #                     print(f"[OOM] Skipping episode in {domain}")
    #                     continue
    #                 raise
    
    #         avg_loss = domain_loss / max(num_episodes, 1)
    #         avg_acc = (total_acc / max(num_episodes, 1)) * 100
    #         avg_dice = total_dice / max(num_episodes, 1)
    
    #         print(
    #             f"Domain: {domain}, Epoch: {epoch+1}/{epochs}, "
    #             f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%, Dice: {avg_dice:.4f}"
    #         )
    
    #         self.schedulers[domain].step()
    
    #     print(f"Consolidating weights for {domain}...")
    #     self.ewc.remember_task(domain, loader, self.device, n_way, k_shot, q_query)
    
        

    
    def evaluate(self, domain, n_way, k_shot, q_query):
        self.model.eval()
    
        if domain not in self.test_loaders:
            return 0.0
    
        def segmentation_metrics(logits, target):
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()
        
            target = target.float()
        
            # Accuracy
            acc = (pred == target).float().mean()
        
            # Dice / F1
            intersection = (pred * target).sum(dim=(1,2,3))
            union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
            dice = (2 * intersection + 1e-6) / (union + 1e-6)
            dice = dice.mean()
        
            # IoU
            iou = (intersection + 1e-6) / (union - intersection + 1e-6)
            iou = iou.mean()
        
            return acc, dice, iou
    
        with torch.no_grad():
            total_acc = 0.0
            total_dice = 0.0
            num_episodes = 0
    
            print(f"Evaluating {domain}...")
            loader = self.test_loaders[domain]
            progress_bar = tqdm(loader, desc=f"{domain} Eval", leave=False)
    
            for batch in progress_bar:
                # -----------------------------
                # UNPACK EPISODE
                # -----------------------------
                img1, img2, mask = batch

                img1 = img1.to(self.device, non_blocking=True)
                img2 = img2.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True).float()
                mask = (mask > 0).float()
                
                if torch.isnan(mask).any():
                    print("NaN in mask")
                
    
                # s_img1 = s_img1.to(self.device)
                # s_img2 = s_img2.to(self.device)
                # s_mask = s_mask.to(self.device)
    
                # q_img1 = q_img1.to(self.device)
                # q_img2 = q_img2.to(self.device)
                # q_mask = q_mask.to(self.device)
    
                # -----------------------------
                # FORWARD
                # -----------------------------
                logits = self.model(img1, img2, domain)
                loss = few_shot_cd_loss(logits, mask)    
                # -----------------------------
                # METRICS
                # -----------------------------
                acc, dice,iou = segmentation_metrics(logits, q_mask)
    
                total_acc += acc.item()
                total_dice += dice.item()
                num_episodes += 1
    
                progress_bar.set_postfix(
                    acc=f"{acc.item() * 100:.2f}%",
                    dice=f"{dice.item():.4f}",
                    iou=f'{iou.item():.4f}'

                )
    
            avg_acc = (total_acc / max(num_episodes, 1)) * 100
            avg_dice = total_dice / max(num_episodes, 1)
    
            print(f"Final metrics on {domain}: accuracy={avg_acc:.2f}%, dice={avg_dice:.4f}")
    
            return avg_dice

        
    def evaluate_all(self, n_way, k_shot, q_query):
        """Evaluate all domains and report Dice scores."""
    
        print("\n--- Evaluating Forgetting Across All Domains ---")
        results = {}
    
        for domain in self.domain_list:
            if domain in self.test_loaders:
                dice = self.evaluate(domain, n_way, k_shot, q_query)
                results[domain] = dice
    
        print("\n=== Summary (Dice Scores) ===")
        for d, score in results.items():
            print(f"{d}: {score:.4f}")
    
        avg = sum(results.values()) / max(len(results), 1)
        print(f"Average Dice: {avg:.4f}")
    
        return results