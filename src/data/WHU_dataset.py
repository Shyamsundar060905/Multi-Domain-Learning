import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision import transforms


img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def has_change(mask, threshold=0.01):
    return (mask.sum() / mask.numel()) > threshold

    
class WHUDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, k_shot=1, q_query=1):
        """
        root_dir/
            train/
                A/
                B/
                OUT/
        """
        self.root = os.path.join(root_dir, split)

        self.transform = img_transform
        self.mask_transform = mask_transform
        self.k_shot = k_shot
        self.q_query = q_query

        self.A_dir = os.path.join(self.root, "A")
        self.B_dir = os.path.join(self.root, "B")
        self.label_dir = os.path.join(self.root, "OUT")

        # ✅ filenames directly aligned
        self.img_names = sorted(os.listdir(self.A_dir))

    def load_triplet(self, idx):
        name = self.img_names[idx]

        img1 = Image.open(os.path.join(self.A_dir, name)).convert("RGB")
        img2 = Image.open(os.path.join(self.B_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.label_dir, name)).convert("L")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = (mask > 0).float()

        return img1, img2, mask

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img1, img2, mask = self.load_triplet(idx)
        return img1, img2, mask

    # def __getitem__(self, idx):
    #     all_indices = list(range(len(self)))
    #     all_indices.remove(idx)

    #     total_needed = self.k_shot + self.q_query

    #     if len(all_indices) >= total_needed:
    #         sampled = random.sample(all_indices, total_needed)
    #     else:
    #         sampled = random.choices(all_indices, k=total_needed)

    #     support_indices = sampled[:self.k_shot]
    #     query_indices = sampled[self.k_shot:]
    #     valid_support = []
        
    #     for i in support_indices:
    #         _, _, m = self.load_triplet(i)
    #         if has_change(m):
    #             valid_support.append(i)
        
    #     # ✅ replace support_indices if we found valid ones
    #     if len(valid_support) > 0:
    #         support_indices = valid_support
    #     if len(support_indices) == 0:
    #         support_indices = [idx]

    #     if len(query_indices) == 0:
    #         query_indices = [idx]
    #     else:
    #         query_indices[0] = idx

    #     s_img1, s_img2, s_mask = [], [], []
    #     q_img1, q_img2, q_mask = [], [], []

    #     for i in support_indices:
    #         i1, i2, m = self.load_triplet(i)
    #         s_img1.append(i1)
    #         s_img2.append(i2)
    #         s_mask.append(m)

    #     for i in query_indices:
    #         i1, i2, m = self.load_triplet(i)
    #         q_img1.append(i1)
    #         q_img2.append(i2)
    #         q_mask.append(m)

    #     return (
    #         torch.stack(s_img1),
    #         torch.stack(s_img2),
    #         torch.stack(s_mask),
    #         torch.stack(q_img1),
    #         torch.stack(q_img2),
    #         torch.stack(q_mask),
    #     )