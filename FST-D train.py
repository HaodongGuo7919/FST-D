import os
import re
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import scipy.io as sio

MODEL_DIR = r"D:\Program Files\JetBrains\PyCharm 2024.1.2\pythonProject"
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from test1 import SwinTransformerSys, StackedSwinSys, flow_smoothness_l1


def compute_refine_detach_alpha(epoch_idx: int, warmup_detach_epochs: int = 15, anneal_epochs: int = 10) -> float:
    if epoch_idx < warmup_detach_epochs:
        return 1.0
    if anneal_epochs <= 0:
        return 0.0
    progress = (epoch_idx - warmup_detach_epochs) / float(anneal_epochs)
    progress = min(max(progress, 0.0), 1.0)
    return 1.0 - progress


class DaWeiYiNewDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.ref_dir = os.path.join(root_dir, "reference_images")
        self.def_dir = os.path.join(root_dir, "deformed_images")
        self.disp_dir = os.path.join(root_dir, "displacement_data")
        self.transform = transform

        if not os.path.isdir(self.disp_dir):
            raise FileNotFoundError(f"displacement_data not found: {self.disp_dir}")

        disp_files = [f for f in os.listdir(self.disp_dir) if f.lower().endswith(".mat")]
        if len(disp_files) == 0:
            raise FileNotFoundError(f"No .mat found in: {self.disp_dir}")

        def _num_key(fn: str) -> int:
            m = re.search(r"(\d+)", fn)
            return int(m.group(1)) if m else -1

        disp_files = sorted(disp_files, key=_num_key)
        self.ids = []
        for f in disp_files:
            n = _num_key(f)
            if n >= 0:
                self.ids.append(str(n))

        if len(self.ids) == 0:
            raise RuntimeError("No valid numbered disp_*.mat files found.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sn = self.ids[idx]

        ref_path = os.path.join(self.ref_dir, f"ref_{sn}.bmp")
        def_path = os.path.join(self.def_dir, f"def_{sn}.bmp")
        disp_path = os.path.join(self.disp_dir, f"disp_{sn}.mat")

        ref_img = Image.open(ref_path).convert("L")
        def_img = Image.open(def_path).convert("L")

        disp_mat = sio.loadmat(disp_path)
        if "uu" not in disp_mat:
            raise KeyError(f"'uu' not found in mat: {disp_path}. Keys={list(disp_mat.keys())}")

        disp = disp_mat["uu"]
        if disp.ndim == 3 and disp.shape[0] == 2:
            pass
        elif disp.ndim == 3 and disp.shape[-1] == 2:
            disp = np.transpose(disp, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected disp shape: {disp.shape} in {disp_path}")

        if self.transform:
            ref_img = self.transform(ref_img)
            def_img = self.transform(def_img)

        disp = torch.tensor(disp, dtype=torch.float32)
        return ref_img, def_img, disp


transform = transforms.Compose([
    transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


class FlowLossSupervised(nn.Module):
    def __init__(
            self,
            w_final_weight=1.0,
            w1_start_weight=0.3,
            w1_mid_weight=0.12,
            w1_end_weight=0.03,
            residual_weight=0.3,
            dw_smoothness_weight=0.002,
            w1_hold_epochs=10,
            w1_fast_decay_ratio=0.35,
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.w_final_w = float(w_final_weight)
        self.w1_start_w = float(w1_start_weight)
        self.w1_mid_w = float(w1_mid_weight)
        self.w1_end_w = float(w1_end_weight)
        self.residual_w = float(residual_weight)
        self.dw_smoothness_w = float(dw_smoothness_weight)
        self.w1_hold_epochs = int(w1_hold_epochs)
        self.w1_fast_decay_ratio = float(w1_fast_decay_ratio)
        self.w1_w = float(w1_start_weight)

    def set_epoch(self, epoch_index: int, total_epochs: int):
        if total_epochs <= 1:
            self.w1_w = self.w1_end_w
            return

        if epoch_index < self.w1_hold_epochs:
            self.w1_w = self.w1_start_w
            return

        remaining_epochs = max(total_epochs - self.w1_hold_epochs - 1, 1)
        progress = float(np.clip((epoch_index - self.w1_hold_epochs) / remaining_epochs, 0.0, 1.0))
        fast_boundary = float(np.clip(self.w1_fast_decay_ratio, 0.05, 0.95))

        if progress <= fast_boundary:
            local_progress = progress / fast_boundary
            self.w1_w = self.w1_start_w + (self.w1_mid_w - self.w1_start_w) * local_progress
        else:
            local_progress = (progress - fast_boundary) / (1.0 - fast_boundary)
            self.w1_w = self.w1_mid_w + (self.w1_end_w - self.w1_mid_w) * local_progress

    def compute_components(self, outputs, target):
        w_final, w1, dw = outputs

        residual_target = target - w1.detach()

        loss_w1 = self.mse(w1, target)
        loss_residual = self.mse(dw, residual_target)
        loss_final = self.mse(w_final, target)
        loss_dw_smooth = flow_smoothness_l1(dw)

        pixel_rmse = torch.sqrt(torch.mean((w_final - target) ** 2))
        total = (
                self.w_final_w * loss_final
                + self.w1_w * loss_w1
                + self.residual_w * loss_residual
                + self.dw_smoothness_w * loss_dw_smooth
        )

        return {
            "total": total,
            "loss_w1": loss_w1.detach(),
            "loss_residual": loss_residual.detach(),
            "loss_final": loss_final.detach(),
            "loss_dw_smooth": loss_dw_smooth.detach(),
            "pixel_rmse": pixel_rmse.detach(),
        }

    def forward(self, outputs, target):
        return self.compute_components(outputs, target)["total"]


def train_with_scheduler(
        model,
        train_loader,
        val_loader,
        criterion: FlowLossSupervised,
        optimizer,
        scheduler,
        num_epochs=50,
        device="cuda",
        save_dir=r"E:\Da_displacement_PTH_new",
        start_epoch=1,
        log_every=100,
):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)

    best_val_final = float("inf")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_idx = epoch - start_epoch
        criterion.set_epoch(epoch_idx, num_epochs)
        refine_detach_alpha = compute_refine_detach_alpha(epoch_idx, warmup_detach_epochs=15, anneal_epochs=10)
        detach_w1_for_refine = refine_detach_alpha >= 1.0 - 1e-8

        model.train()
        total_loss = 0.0

        running_loss = 0.0
        running_w1 = 0.0
        running_residual = 0.0
        running_final = 0.0
        running_dw_smooth = 0.0
        running_rmse = 0.0
        running_n = 0

        for i, (img1, img2, disp_gt) in enumerate(train_loader):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            disp_gt = disp_gt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(
                img1, img2,
                detach_w1_for_refine=detach_w1_for_refine,
                refine_detach_alpha=refine_detach_alpha,
            )
            metrics = criterion.compute_components(outputs, disp_gt)
            loss = metrics["total"]

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            running_loss += float(loss.item())
            running_w1 += float(metrics["loss_w1"].item())
            running_residual += float(metrics["loss_residual"].item())
            running_final += float(metrics["loss_final"].item())
            running_dw_smooth += float(metrics["loss_dw_smooth"].item())
            running_rmse += float(metrics["pixel_rmse"].item())
            running_n += 1

            if log_every is not None and log_every > 0 and running_n == log_every:
                print(
                    f"Epoch [{epoch}] Batch [{i + 1}/{len(train_loader)}] "
                    f"loss={running_loss / running_n:.6f} "
                    f"loss_w1={running_w1 / running_n:.6f} "
                    f"loss_residual={running_residual / running_n:.6f} "
                    f"loss_final={running_final / running_n:.6f} "
                    f"loss_dw_smooth={running_dw_smooth / running_n:.6f} "
                    f"pixel_RMSE={running_rmse / running_n:.6f} "
                    f"w1_weight={criterion.w1_w:.4f} "
                    f"detach_refine={detach_w1_for_refine} "
                    f"refine_alpha={refine_detach_alpha:.4f}"
                )
                running_loss, running_w1, running_residual, running_final, running_dw_smooth, running_rmse, running_n = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

        scheduler.step()
        avg_train_loss = total_loss / max(1, len(train_loader))

        model.eval()
        val_loss, val_w1, val_residual, val_final, val_dw_smooth, val_rmse = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for img1, img2, disp_gt in val_loader:
                img1 = img1.to(device, non_blocking=True)
                img2 = img2.to(device, non_blocking=True)
                disp_gt = disp_gt.to(device, non_blocking=True)

                outputs = model(
                    img1, img2,
                    detach_w1_for_refine=detach_w1_for_refine,
                    refine_detach_alpha=refine_detach_alpha,
                )
                metrics = criterion.compute_components(outputs, disp_gt)
                val_loss += float(metrics["total"].item())
                val_w1 += float(metrics["loss_w1"].item())
                val_residual += float(metrics["loss_residual"].item())
                val_final += float(metrics["loss_final"].item())
                val_dw_smooth += float(metrics["loss_dw_smooth"].item())
                val_rmse += float(metrics["pixel_rmse"].item())

        num_val_batches = max(1, len(val_loader))
        avg_val_loss = val_loss / num_val_batches
        avg_val_w1 = val_w1 / num_val_batches
        avg_val_residual = val_residual / num_val_batches
        avg_val_final = val_final / num_val_batches
        avg_val_dw_smooth = val_dw_smooth / num_val_batches
        avg_val_rmse = val_rmse / num_val_batches

        print(
            f"Epoch [{epoch}] train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f} "
            f"val_loss_w1={avg_val_w1:.6f} val_loss_residual={avg_val_residual:.6f} "
            f"val_loss_final={avg_val_final:.6f} val_loss_dw_smooth={avg_val_dw_smooth:.6f} "
            f"val_pixel_RMSE={avg_val_rmse:.6f} w1_weight={criterion.w1_w:.4f} "
            f"detach_refine={detach_w1_for_refine} refine_alpha={refine_detach_alpha:.4f} "
            f"lr={scheduler.get_last_lr()}"
        )

        if avg_val_final < best_val_final:
            best_val_final = avg_val_final
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_stacked_model_epoch_{epoch}.pth"))
            print(f"Best stacked model saved at epoch {epoch}")

        torch.save(model.state_dict(), os.path.join(save_dir, f"stacked_epoch_{epoch}.pth"))


if __name__ == "__main__":
    DATA_ROOT = r"D:\DaWeiYiDataset"

    full_dataset = DaWeiYiNewDataset(root_dir=DATA_ROOT, transform=transform)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    batch_size = 16
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    save_dir = r"E:\Da_displacement_PTH_new5"
    os.makedirs(save_dir, exist_ok=True)

    model = StackedSwinSys(BaseSwinSys=SwinTransformerSys)
    criterion = FlowLossSupervised(
        w1_start_weight=0.3, w1_mid_weight=0.12, w1_end_weight=0.03,
        residual_weight=0.3, w1_hold_epochs=10, w1_fast_decay_ratio=0.35,
    )

    pretrained_model_path = r"E:\Da_displacement_PTH_new2\best_stacked_model_epoch_200.pth"
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"))
        print("Pre-trained stacked model loaded successfully.")
        start_epoch = 12
    else:
        print("Pre-trained stacked model not found. Training from scratch.")
        start_epoch = 1

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    if start_epoch > 1:
        scheduler.step(start_epoch - 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_with_scheduler(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=265, device=device, save_dir=save_dir, start_epoch=start_epoch, log_every=100,
    )