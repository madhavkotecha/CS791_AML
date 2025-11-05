### Add necessary imports ###
import numpy as np
from acquisition_functions import expected_improvement, probability_of_improvement
from kernels import rbf_kernel, matern_kernel, rational_quadratic_kernel
from train_test import train_and_test_NN, train_and_test_CNN
from utils import gaussian_process_predict, optimize_hyperparameters, seed_everything
import argparse
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt

NN_SEARCH_SPACE = {
    'hidden_size': [100, 200, 300, 400, 500],
    'training_epochs': [1, 10],
    'lr': [1e-5, 1e-1],
    'batch_size': [16, 32, 64, 128, 256],
    'dropout_rate': [0.0, 0.5],
    'weight_decay': [1e-6, 1e-2]
}

CNN_SEARCH_SPACE = {
    'lr': [1e-4, 1e-2], 
    'batch_size': [32, 64, 128, 256], 
    'training_epochs': [5, 20], 
    'optimizer': ['adam', 'rmsprop'] 
}

CNN_FIXED_PARAMS = {
    'hidden_size': 256, 
    'dropout_rate': 0.0, 
    'weight_decay': 1e-4 
}

def random_hyperparams(model_type):
    
    if model_type == 'nn':
        current_space = NN_SEARCH_SPACE
        hyperparams = {
            'hidden_size': int(np.random.choice(current_space['hidden_size'])),
            'training_epochs': int(np.random.randint(current_space['training_epochs'][0], current_space['training_epochs'][1] + 1)),
            'lr': float(10**np.random.uniform(np.log10(current_space['lr'][0]), np.log10(current_space['lr'][1]))),
            'batch_size': int(np.random.choice(current_space['batch_size'])),
            'dropout_rate': float(np.random.uniform(current_space['dropout_rate'][0], current_space['dropout_rate'][1])),
            'weight_decay': float(10**np.random.uniform(np.log10(current_space['weight_decay'][0]), np.log10(current_space['weight_decay'][1])))
        }
    else:
        current_space = CNN_SEARCH_SPACE
        hyperparams = {
            'lr': float(10**np.random.uniform(np.log10(current_space['lr'][0]), np.log10(current_space['lr'][1]))),
            'batch_size': int(np.random.choice(current_space['batch_size'])),
            'training_epochs': int(np.random.randint(current_space['training_epochs'][0], current_space['training_epochs'][1] + 1)),
            'optimizer': np.random.choice(current_space['optimizer'])
        }
        hyperparams.update(CNN_FIXED_PARAMS)

    return hyperparams

def dict2arr(hyperparams, model_type):
    
    if model_type == 'nn':
        return np.array([
            hyperparams["hidden_size"],
            hyperparams["training_epochs"],
            np.log10(hyperparams["lr"]),
            hyperparams["batch_size"],
            hyperparams["dropout_rate"],
            np.log10(hyperparams["weight_decay"])
        ])
    optimizer_map = {'adam': 0, 'rmsprop': 1}
    return np.array([
        np.log10(hyperparams["lr"]),
        hyperparams["batch_size"],
        hyperparams["training_epochs"],
        optimizer_map[hyperparams["optimizer"]]
    ])


def arr2dict(arr, model_type):
        
    if model_type == 'nn':
        current_space = NN_SEARCH_SPACE
        hsize_diff = [abs(x - arr[0]) for x in current_space['hidden_size']]
        hidden_size = int(current_space['hidden_size'][np.argmin(hsize_diff)])
        
        training_epochs = int(np.clip(np.round(arr[1]), current_space['training_epochs'][0], current_space['training_epochs'][1]))
        lr = float(10 ** np.clip(arr[2], np.log10(current_space['lr'][0]), np.log10(current_space['lr'][1])))
        
        bsize_diff = [abs(x - arr[3]) for x in current_space['batch_size']]
        batch_size = int(current_space['batch_size'][np.argmin(bsize_diff)])
        
        dropout_rate = float(np.clip(arr[4], current_space['dropout_rate'][0], current_space['dropout_rate'][1]))
        weight_decay = float(10 ** np.clip(arr[5], np.log10(current_space['weight_decay'][0]), np.log10(current_space['weight_decay'][1])))
        
        return {
            "hidden_size": hidden_size,
            "training_epochs": training_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "weight_decay": weight_decay
        }
    else:
        current_space = CNN_SEARCH_SPACE
        hyperparams = CNN_FIXED_PARAMS.copy()
        
        hyperparams['lr'] = float(10 ** np.clip(arr[0], np.log10(current_space['lr'][0]), np.log10(current_space['lr'][1])))
        
        bsize_diff = [abs(x - arr[1]) for x in current_space['batch_size']]
        hyperparams['batch_size'] = int(current_space['batch_size'][np.argmin(bsize_diff)])
        
        hyperparams['training_epochs'] = int(np.clip(np.round(arr[2]), current_space['training_epochs'][0], current_space['training_epochs'][1]))

        opt_map = current_space['optimizer']
        optimizer_index = np.clip(int(np.round(arr[3])), 0, len(opt_map) - 1)
        hyperparams['optimizer'] = opt_map[optimizer_index]

        return hyperparams


def candidate_points(n_candidates=1000, model_type='nn'):
    candidates = []
    
    for i in range(n_candidates):
        hyperparams = random_hyperparams(model_type)
        candidates.append(dict2arr(hyperparams, model_type))
    
    return np.array(candidates)


def parse_args():
    parser = argparse.ArgumentParser(description='Train and Test Models with Hyperparameters')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model_type', type=str, choices=['nn', 'cnn'], default='nn', help='Type of model to use')
    parser.add_argument('--acquisition_function', type=str, choices=['ei', 'pi'], default='ei', help='Acquisition function to use')
    parser.add_argument('--kernel', type=str, choices=['rbf', 'matern', 'rational_quadratic'], default='rbf', help='Kernel function to use')
    parser.add_argument('--max_budget', type=int, default=25, help='Maximum budget for hyperparameter optimization')
    parser.add_argument('--init_points', type=int, default=10, help='Number of initial random points for hyperparameter optimization')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)

    assert args.max_budget >= args.init_points, "max_budget should be greater than init_points"

    if args.kernel == 'rbf':
        kernel_func = rbf_kernel
    elif args.kernel == 'matern':
        kernel_func = matern_kernel
    elif args.kernel == 'rational_quadratic':
        kernel_func = rational_quadratic_kernel

    if args.acquisition_function == 'ei':
        acquisition_func = expected_improvement
    elif args.acquisition_function == 'pi':
        acquisition_func = probability_of_improvement

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_validation_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_size = int(0.8 * len(train_validation_dataset))
    validation_size = len(train_validation_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(train_validation_dataset, [train_size, validation_size])
    train_val_datasets = (train_dataset, validation_dataset) # Give this as input to train_and_test_NN or train_and_test_CNN functions

    # Perform 'init_points' initial random hyperparameter sampling from the hyperparameter space
    # hyperparams = {
    #     "batch_size": 64,
    #     "training_epochs": 10,
    #     "hidden_size": 128,
    #     "dropout_rate": 0.001,
    #     # optimization
    #     "lr": 0.01,
    #     "weight_decay": 0.01
    # }
    X_observed = []
    y_observed = []
    all_accuracies = []
    best_accuracy = -np.inf
    best_hyperparams = None

    for i in range(args.init_points):
        hyperparams = random_hyperparams(args.model_type)
        print("Hyper:", hyperparams)
        if args.model_type == 'nn':
            curr_accuracy = train_and_test_NN(train_val_datasets, hyperparams, seed=args.seed)
        else:
            curr_accuracy = train_and_test_CNN(train_val_datasets, hyperparams, seed=args.seed)
        
        X_observed.append(dict2arr(hyperparams, args.model_type))
        y_observed.append(curr_accuracy)
        all_accuracies.append(curr_accuracy)
        
        print(f"Accuracy: {curr_accuracy:.2f}%")

        if curr_accuracy > best_accuracy:
            best_accuracy = curr_accuracy
            best_hyperparams = hyperparams
            print(f"New best accuracy: {best_accuracy:.2f}%")

    print("After Initial Random Sampling")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"Best Hyperparameters: {best_hyperparams}")


    X_observed = np.array(X_observed) # [10, 6]
    y_observed = np.array(y_observed) # [10]

    for step in range(args.max_budget - args.init_points):

        # min-max normalization
        X_min = np.min(X_observed, axis=0)
        X_max = np.max(X_observed, axis=0)
        X_observed_norm = (X_observed - X_min) / (X_max - X_min + 1e-8)
        
        length_scale, sigma_f = optimize_hyperparameters(X_observed_norm, y_observed, kernel_func)
        print(f"Optimized GP hyperparameters - length_scale: {length_scale:.4f}, sigma_f: {sigma_f:.4f}")
        
        candidates = candidate_points(n_candidates=1000, model_type=args.model_type)
        candidates_norm = (candidates - X_min) / (X_max - X_min + 1e-8)
        
        # changed: pass now normalized 
        mu, sigma = gaussian_process_predict(X_observed_norm, y_observed, candidates_norm, kernel_func, length_scale=length_scale, sigma_f=sigma_f)
        acquisition_values = acquisition_func(mu, sigma, np.max(y_observed))
        
        best_candidate_idx = np.argmax(acquisition_values)
        next_point = candidates[best_candidate_idx]
        next_hyperparams = arr2dict(next_point, args.model_type)

        print(f"Next hyperparameters: {next_hyperparams}")
        print(f"Predicted mean: {mu[best_candidate_idx][0]:.2f}, std: {sigma[best_candidate_idx][0]:.2f}")
        print(f"Acquisition value: {acquisition_values[best_candidate_idx][0]:.4f}")
        
        if args.model_type == 'nn':
            accuracy = train_and_test_NN(train_val_datasets, next_hyperparams, seed=args.seed)
        else:
            accuracy = train_and_test_CNN(train_val_datasets, next_hyperparams, seed=args.seed)
        
        print(f"Mean Validation Accuracy: {accuracy:.2f}%")
        
        X_observed = np.vstack([X_observed, next_point])
        y_observed = np.append(y_observed, accuracy)
        all_accuracies.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hyperparams = next_hyperparams
            print(f"New best accuracy: {best_accuracy:.2f}%")

    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(all_accuracies)+1), all_accuracies, marker='o')
    plt.title(f'Validation Accuracy Progression ({args.kernel} + {args.acquisition_function})')
    plt.xlabel('Iteration')
    plt.ylabel('Validation Accuracy (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.kernel}_{args.acquisition_function}_{args.model_type}_N{args.max_budget}.png")

    # full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    final_datasets = (train_dataset, test_dataset)

    print(f"{args.model_type} - {args.kernel} - {args.acquisition_function}, best hyperparams: {best_hyperparams}")
    if args.model_type == 'nn':
        test_accuracy = train_and_test_NN(final_datasets, best_hyperparams, seed=args.seed)
    else:
        test_accuracy = train_and_test_CNN(final_datasets, best_hyperparams, seed=args.seed)
    
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")


    # mean_validation_accuracy = train_and_test_NN(datasets=train_val_datasets, hyperparams=hyperparams)
    # print(mean_validation_accuracy)
    print(f"All validation accuracies: {all_accuracies}")

    '''
    python main.py --model_type nn 
    '''