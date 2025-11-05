# CS791 Assignment 4: Hyperparameter Tuning using Gaussian Process Surrogates

## 🗂️ Data Setup

Data will be automatically downloaded when `main.py` is run.

### Prerequisites

#### Option 1: Conda Environment (Recommended)
```bash
# Create a new conda environment
conda create -n cs791_a4 python=3.10

# Activate the environment
conda activate cs791_a4

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

## 🎯 Task 1: Example Workflow

```bash
# Run main.py to execute the entire BO process
python main.py --model_type nn --acquisition_function pi --kernel matern --max_budget 25 --init_points 10
```

## 🎯 Task 2: Example Workflow

```bash
# Run main.py to execute the entire BO process
python main.py --model_type cnn --acquisition_function pi --kernel rational_quadratic --max_budget 50 --init_points 10
```
