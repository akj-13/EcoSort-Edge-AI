"""
EcoSort: compact six-class waste classifier.
Run: python train_template.py --epochs 60 --batch_size 32 --lr 0.001
Expected dataset layout: data/public/{train,val}/{cardboard,glass,metal,paper,plastic,trash}/...
"""
import argparse, os, random, copy, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, kernel=3, stride=1, groups=1, act=True):
        layers = [nn.Conv2d(cin, cout, kernel, stride, kernel // 2, groups=groups, bias=False),
                  nn.BatchNorm2d(cout)]
        if act: layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)

class MBConv(nn.Module):
    def __init__(self, cin, cout, stride=1, expand=2):
        super().__init__()
        hidden = cin * expand
        layers = []
        if expand != 1: layers.append(ConvBNAct(cin, hidden, kernel=1))
        layers += [ConvBNAct(hidden, hidden, kernel=3, stride=stride, groups=hidden),
                   ConvBNAct(hidden, cout, kernel=1, act=False)]
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and cin == cout
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        y = self.block(x)
        if self.use_residual: y = y + x
        return self.act(y)

class EcoSortNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 24, stride=2),       # 48 x 48
            MBConv(24, 32, stride=2),         # 24 x 24
            MBConv(32, 32, stride=1),
            MBConv(32, 48, stride=2),         # 12 x 12
            MBConv(48, 48, stride=1),
            MBConv(48, 64, stride=2),         # 6 x 6
            MBConv(64, 64, stride=1),
            MBConv(64, 96, stride=2),         # 3 x 3
            ConvBNAct(96, 128, kernel=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.20), nn.Linear(128, num_classes))
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def make_loaders(data_dir, batch_size, workers):
    train_dir, val_dir = os.path.join(data_dir, "train"), os.path.join(data_dir, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Expected {train_dir} and {val_dir}. Extract the supplied dataset into data/public/.")
    train_tf = transforms.Compose([
        transforms.Resize((104, 104)),
        transforms.RandomResizedCrop(96, scale=(0.78, 1.0), ratio=(0.90, 1.10)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.025),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    if train_ds.classes != CLASSES or val_ds.classes != CLASSES:
        raise ValueError(f"Class folder order must be {CLASSES}; found train={train_ds.classes}, val={val_ds.classes}")
    pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    return train_dl, val_dl

def export_onnx(model, output_path):
    model = model.cpu().eval()
    dummy = torch.zeros(1, 3, 96, 96, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, output_path, export_params=True, opset_version=17,
        do_constant_folding=True, input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False
    )
    print(f"ONNX exported: {output_path} ({os.path.getsize(output_path)/1024/1024:.3f} MiB)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/public")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--output", default="model.onnx")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=12)
    args = p.parse_args()
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_dl, val_dl = make_loaders(args.data_dir, args.batch_size, args.workers)
    model = EcoSortNet().to(device)
    nparams = count_parameters(model)
    print(f"Device: {device}; train={len(train_dl.dataset)}; val={len(val_dl.dataset)}; params={nparams:,}")
    if nparams > 350_000: raise RuntimeError("Parameter budget exceeded")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc, best_state, stale = 0.0, None, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = correct = seen = 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x); loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item(); seen += y.size(0)
        model.eval(); vc = vn = 0
        with torch.no_grad():
            for x, y in val_dl:
                logits = model(x.to(device, non_blocking=True))
                vc += (logits.argmax(1).cpu() == y).sum().item(); vn += y.size(0)
        scheduler.step()
        train_acc, val_acc = 100*correct/seen, 100*vc/vn
        print(f"Epoch {epoch:03d}/{args.epochs} loss={total_loss/seen:.4f} train={train_acc:.2f}% val={val_acc:.2f}%", flush=True)
        if val_acc > best_acc:
            best_acc, stale = val_acc, 0
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(best_state, "best_model.pth")
        else:
            stale += 1
        if stale >= args.patience:
            print(f"Early stopping; best validation accuracy {best_acc:.2f}%")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best validation accuracy: {best_acc:.2f}% (local validation only; hidden-test accuracy may differ)")
    export_onnx(model, args.output)
    print("Run: python verify_submission.py --model model.onnx --val_dir data/public/val")
    print(f"Elapsed: {(time.time()-start)/60:.1f} minutes")

if __name__ == "__main__":
    main()
