# EcoSort — Edge-AI Vision Challenge

Six-class image classification at 96×96 RGB: cardboard, glass, metal, paper, plastic, trash.

## Dataset setup
Extract the official public dataset so the structure is:
```
data/public/train/<class>/
data/public/val/<class>/
```
The supplied archive is `data/ecosort_dataset_public.zip`. With the starter kit, run `python starter_kit/download_data.py` from the repository root if the `data/public` folders are not already extracted.

## Environment
Python 3.10 or 3.11 is recommended. Install the provided dependencies:
```
python -m pip install -r starter_kit/requirements.txt
```
Use a PyTorch build compatible with your machine. On Windows, start with `num_workers=0` if DataLoader worker startup causes problems.

## Train and export
From the repository root:
```
python starter_kit/train_template.py --data_dir data/public --epochs 60 --batch_size 32 --lr 0.001 --output model.onnx
```
The script trains a compact depthwise-separable convolutional network with residual blocks, mild image augmentation, AdamW, cosine learning-rate decay, and early stopping on the provided validation split. It exports the best validation checkpoint.

## Constraints
- Input: float32 `(batch, 3, 96, 96)` normalized using ImageNet mean/std.
- Output: float32 `(batch, 6)` logits ordered as `cardboard, glass, metal, paper, plastic, trash`.

## Reproducibility and process notes
- Random seed: 42.
- The model architecture is in `starter_kit/train_template.py`.
