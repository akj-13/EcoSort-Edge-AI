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

## Verify
```
python starter_kit/verify_submission.py --model model.onnx --val_dir data/public/val
```
The verifier checks file size, ONNX initializer parameter count, input/output signatures, CPU inference, and public validation accuracy. Keep the output and record the actual results in your submission notes.

## Constraints
- Input: float32 `(batch, 3, 96, 96)` normalized using ImageNet mean/std.
- Output: float32 `(batch, 6)` logits ordered as `cardboard, glass, metal, paper, plastic, trash`.
- Maximum 350,000 parameters.
- Maximum 1.50 MiB file size.
- Target hidden-test accuracy: at least 75%. Public validation accuracy is only a proxy; it does not guarantee hidden-test performance.

## Reproducibility and process notes
- Random seed: 42.
- The model architecture is in `starter_kit/train_template.py`.
- Training uses only the provided training split and monitors the provided validation split.
- Record the command, device, best validation accuracy, parameter count, ONNX file size, and any changes you make. Do not claim a score you did not measure.

## Suggested repository contents
Commit the starter kit, model training/export code, README, and a small text file containing actual verification results. Avoid committing the full dataset unless the organizers explicitly request it. If model.onnx is requested separately by the form, upload it there.
