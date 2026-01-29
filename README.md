# Vision Mamba (Vim)

Implementation of a Vision Mamba (Vim) model in PyTorch, featuring:

- Selective State Space Model (SSM) core with a parallel associative scan
- Bidirectional sequence processing for 2D images
- CIFAR-100 training pipeline and benchmarking against a DeiT-Tiny ViT baseline

## Project Structure

- `vim/`: Core model components (patch embedding, scan, SSM, Vim blocks, full model)
- `data/`: Dataset loaders (CIFAR-100 and Tiny-ImageNet)
- `utils/`: Metrics and visualization utilities
- `benchmarks/`: Memory and baseline benchmarks
- `train.py`: Main training script
- `evaluate.py`: Evaluation script

## Setup

```bash
pip install -r requirements.txt
```

## Training on CIFAR-100

```bash
python train.py \
  --dataset cifar100 \
  --data-dir /path/to/cifar \
  --epochs 200 \
  --batch-size 128 \
  --model vim_tiny
```

## Evaluation

```bash
python evaluate.py \
  --dataset cifar100 \
  --data-dir /path/to/cifar \
  --checkpoint /path/to/checkpoint.pth
```

## Benchmarks

- Memory benchmark utilities live in `benchmarks/memory_benchmark.py`
- DeiT-Tiny baseline is provided in `benchmarks/vit_baseline.py`

