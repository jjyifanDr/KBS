"""
Parameter Sensitivity Analysis for KERN (Paper Section 4.5)

Analyzes 4 key hyperparameters:
1. Evidence space dimension (d_e): [4, 8, 16, 32, 64]
2. Rule activation threshold (tau): [0.01, 0.05, 0.1, 0.3, 0.5]
3. Conflict detection threshold (eta): [0.1, 0.3, 0.5, 0.7, 0.9]
4. Conservatism coefficient (gamma): [0.2, 0.4, 0.6, 0.8, 1.0]

Each parameter is varied while others are fixed at default values.
2 runs per configuration for statistical stability.

Outputs:
- sensitivity_results/sensitivity_combined.tiff (2x2 combined figure)
- sensitivity_results/sensitivity_optimal_table.txt (最优值表格)
- sensitivity_results/sensitivity_all_summary.txt (所有结果汇总)
- Tables printed to console
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from KERN import (
    Config, KERN, KERNLoss, train_stage1, train_stage2, evaluate_model,
    load_dataset, set_seed, get_knowledge_base, RuleEncoder
)


# =============================================================================
# Parameter Sensitivity Runner
# =============================================================================

def run_parameter_experiment(dataset_name, param_name, param_value, config, device='cuda', n_runs=2):
    """
    Run experiment with a specific parameter value.
    n_runs: number of runs for stability.
    """
    auc_roc_list = []
    auc_pr_list = []
    f1_list = []
    
    for run_id in range(n_runs):
        set_seed(42 + run_id * 10)
        
        X, y, feature_names = load_dataset(dataset_name, config.data_dir)
        if X is None:
            return None
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        if len(np.unique(y)) < 2:
            return None
        
        unique, counts = np.unique(y, return_counts=True)
        use_stratify = all(counts >= 2)
        
        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_t = torch.FloatTensor(X_train_scaled)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val_scaled)
        y_val_t = torch.FloatTensor(y_val)
        X_test_t = torch.FloatTensor(X_test_scaled)
        y_test_t = torch.FloatTensor(y_test)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        input_dim = X_train.shape[1]
        tau = Config.get_tau_for_dataset(dataset_name)
        
        # Build model with specific parameter
        if param_name == 'd_e':
            model = KERN(input_dim, d_e=int(param_value), tau=tau, 
                        eta=config.eta, gamma=config.gamma)
        elif param_name == 'tau':
            model = KERN(input_dim, d_e=config.d_e, tau=float(param_value),
                        eta=config.eta, gamma=config.gamma)
        elif param_name == 'eta':
            model = KERN(input_dim, d_e=config.d_e, tau=tau,
                        eta=float(param_value), gamma=config.gamma)
        elif param_name == 'gamma':
            model = KERN(input_dim, d_e=config.d_e, tau=tau,
                        eta=config.eta, gamma=float(param_value))
        else:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        rule_encoder = RuleEncoder('all-MiniLM-L6-v2', device=device, seed=42)
        knowledge_base = get_knowledge_base(dataset_name)
        model.set_rule_encoder(rule_encoder)
        model.set_knowledge_base(knowledge_base)
        
        model, _, _ = train_stage1(
            model, train_loader, val_loader,
            knowledge_base, rule_encoder,
            epochs=config.epochs_stage1,
            lr=config.lr,
            device=device
        )
        model = train_stage2(
            model, train_loader, val_loader,
            epochs=config.epochs_stage2,
            lr=config.lr,
            lambda_c=config.lambda_c,
            lambda_r=config.lambda_r,
            device=device
        )
        
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(device)
                labels = batch[1].to(device)
                anomaly_score, _ = model(x)
                all_preds.extend(anomaly_score.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        if len(np.unique(all_labels)) < 2:
            auc_roc_list.append(0.5)
            auc_pr_list.append(0.0)
            f1_list.append(0.0)
        else:
            auc_roc = roc_auc_score(all_labels, all_preds)
            precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_preds)
            auc_pr = auc(recall_curve, precision_curve)
            
            best_f1 = 0
            for threshold in np.linspace(0, 1, 100):
                pred_binary = (all_preds > threshold).astype(int)
                f1 = f1_score(all_labels, pred_binary, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
            
            auc_roc_list.append(auc_roc)
            auc_pr_list.append(auc_pr)
            f1_list.append(best_f1)
    
    return {
        'auc_roc': np.mean(auc_roc_list),
        'auc_pr': np.mean(auc_pr_list),
        'f1_score': np.mean(f1_list),
        'auc_roc_std': np.std(auc_roc_list),
        'auc_pr_std': np.std(auc_pr_list),
        'f1_std': np.std(f1_list)
    }


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_combined_sensitivity_curves(results_dict, param_configs, output_path):
    """
    Plot all 4 sensitivity analysis curves in a 2x2 combined figure.
    """
    # Set up the figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Flatten axes for easy indexing
    axes = axes.flatten()
    
    # Define subplot labels
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    
    # Colors for each subplot (using a professional color palette)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Parameter-specific x-axis labels
    x_labels = {
        'd_e': r'$d_e$',
        'tau': r'$\tau$',
        'eta': r'$\eta$',
        'gamma': r'$\gamma$'
    }
    
    for idx, (param_name, param_info) in enumerate(param_configs.items()):
        ax = axes[idx]
        param_values = param_info['values']
        param_label = param_info['label']
        
        # Extract data
        dataset = 'wdbc'
        values = []
        stds = []
        
        for val in param_values:
            key = f"{param_name}_{val}"
            if dataset in results_dict and key in results_dict[dataset]:
                result = results_dict[dataset][key]
                if result is not None:
                    values.append(result['auc_roc'])
                    stds.append(result.get('auc_roc_std', 0.0))
                else:
                    values.append(np.nan)
                    stds.append(0.0)
            else:
                values.append(np.nan)
                stds.append(0.0)
        
        # Convert to numpy arrays for plotting
        values = np.array(values)
        stds = np.array(stds)
        
        # Create x positions
        x_positions = np.arange(len(param_values))
        
        # Plot main line with error bars
        ax.plot(x_positions, values, 'o-', color=colors[idx], 
                linewidth=2.5, markersize=9, label='WDBC')
        
        # Add error bars (standard deviation)
        ax.errorbar(x_positions, values, yerr=stds, fmt='none', 
                   color=colors[idx], capsize=4, alpha=0.5)
        
        # Add confidence band (filled between)
        ax.fill_between(x_positions, 
                        values - stds,
                        values + stds,
                        color=colors[idx], alpha=0.15)
        
        # Set x-axis labels
        ax.set_xlabel(x_labels[param_name], fontsize=14, fontname='Times New Roman')
        ax.set_ylabel('AUC-ROC', fontsize=14, fontname='Times New Roman')
        
        # Set x-ticks
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(v) for v in param_values], fontsize=14, fontname='Times New Roman')
        
        # Set y-ticks
        ax.tick_params(axis='y', labelsize=14)
        
        # Set subplot title
        ax.set_title(f'{subplot_labels[idx]} {param_label}', fontsize=14, fontname='Times New Roman')
        
        # Add legend
        ax.legend(fontsize=14, prop={'family': 'Times New Roman'})
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits
        ax.set_ylim(0.85, 1.01)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the combined figure
    plt.savefig(output_path, dpi=150, format='tiff', bbox_inches='tight')
    plt.close()
    print(f"  Saved combined sensitivity plot to {output_path}")


def print_optimal_table(all_results, param_configs):
    """Print optimal parameter values table."""
    print("\n" + "=" * 60)
    print("OPTIMAL PARAMETER VALUES ON WDBC")
    print("=" * 60)
    
    print("\n" + "-" * 70)
    print(f"{'Parameter':<20} {'Default':>12} {'Optimal':>12} {'AUC-ROC at Optimal':>20}")
    print("-" * 70)
    
    dataset = 'wdbc'
    optimal_results = {}
    
    for param_name, param_info in param_configs.items():
        default_val = param_info['default']
        param_values = param_info['values']
        
        best_val = None
        best_auc = -1
        for val in param_values:
            key = f"{param_name}_{val}"
            if dataset in all_results and key in all_results[dataset]:
                result = all_results[dataset][key]
                if result is not None:
                    auc_val = result['auc_roc']
                    if auc_val > best_auc:
                        best_auc = auc_val
                        best_val = val
        
        if best_val is not None:
            print(f"{param_info['label']:<20} {str(default_val):>12} {str(best_val):>12} {best_auc:>19.4f}")
            optimal_results[param_name] = {'optimal': best_val, 'auc': best_auc}
        else:
            print(f"{param_info['label']:<20} {str(default_val):>12} {'N/A':>12} {'N/A':>20}")
    
    print("-" * 70)
    
    # Also print a summary of the performance range
    print("\n" + "=" * 60)
    print("PERFORMANCE STABILITY SUMMARY")
    print("=" * 60)
    print("\n" + "-" * 70)
    print(f"{'Parameter':<20} {'Min AUC-ROC':>15} {'Max AUC-ROC':>15} {'Range':>15}")
    print("-" * 70)
    
    for param_name, param_info in param_configs.items():
        param_values = param_info['values']
        auc_values = []
        for val in param_values:
            key = f"{param_name}_{val}"
            if dataset in all_results and key in all_results[dataset]:
                result = all_results[dataset][key]
                if result is not None:
                    auc_values.append(result['auc_roc'])
        
        if auc_values:
            min_auc = min(auc_values)
            max_auc = max(auc_values)
            range_auc = max_auc - min_auc
            print(f"{param_info['label']:<20} {min_auc:>15.4f} {max_auc:>15.4f} {range_auc:>15.4f}")
    
    print("-" * 70)
    
    return optimal_results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("KERN Parameter Sensitivity Analysis (Section 4.5)")
    print("=" * 60)
    
    set_seed(42)
    
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Running on dataset: WDBC (only)")
    
    # Parameter definitions
    param_configs = {
        'd_e': {
            'values': [4, 8, 16, 32, 64],
            'label': 'Evidence Space Dimension ($d_e$)',
            'default': 16,
            'file_suffix': 'd_e'
        },
        'tau': {
            'values': [0.01, 0.05, 0.1, 0.3, 0.5],
            'label': 'Rule Activation Threshold ($\\tau$)',
            'default': 0.05,
            'file_suffix': 'tau'
        },
        'eta': {
            'values': [0.1, 0.3, 0.5, 0.7, 0.9],
            'label': 'Conflict Detection Threshold ($\\eta$)',
            'default': 0.5,
            'file_suffix': 'eta'
        },
        'gamma': {
            'values': [0.2, 0.4, 0.6, 0.8, 1.0],
            'label': 'Conservatism Coefficient ($\\gamma$)',
            'default': 0.8,
            'file_suffix': 'gamma'
        }
    }
    
    datasets = ['wdbc']  # Only WDBC for parameter sensitivity
    
    results_dir = './sensitivity_results'
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset}")
        print(f"{'='*50}")
        
        all_results[dataset] = {}
        
        for param_name, param_info in param_configs.items():
            param_values = param_info['values']
            param_label = param_info['label']
            
            print(f"\n  Analyzing {param_label}...")
            
            for val in param_values:
                print(f"    Value: {val}")
                result = run_parameter_experiment(
                    dataset, param_name, val, config, device, n_runs=2
                )
                
                key = f"{param_name}_{val}"
                all_results[dataset][key] = result
                
                if result is not None:
                    print(f"      AUC-ROC: {result['auc_roc']:.4f} ± {result.get('auc_roc_std', 0):.4f}")
    
    # Print summary tables
    print("\n" + "=" * 60)
    print("PARAMETER SENSITIVITY SUMMARY")
    print("=" * 60)
    
    for param_name, param_info in param_configs.items():
        param_values = param_info['values']
        param_label = param_info['label']
        
        print(f"\n{param_label}:")
        print("-" * 50)
        print(f"{'Value':>10} {'WDBC':>15}")
        print("-" * 50)
        
        for val in param_values:
            key = f"{param_name}_{val}"
            if 'wdbc' in all_results and key in all_results['wdbc']:
                result = all_results['wdbc'][key]
                if result is not None:
                    print(f"{val:>10} {result['auc_roc']:>14.4f}")
                else:
                    print(f"{val:>10} {'N/A':>14}")
            else:
                print(f"{val:>10} {'N/A':>14}")
        
        print("-" * 50)
    
    # Print optimal values table
    optimal_results = print_optimal_table(all_results, param_configs)
    
    # Save all results
    summary_path = os.path.join(results_dir, 'sensitivity_all_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("KERN PARAMETER SENSITIVITY ANALYSIS RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        for param_name, param_info in param_configs.items():
            param_values = param_info['values']
            param_label = param_info['label']
            
            f.write(f"\n{param_label}:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Value':>10} {'WDBC':>15}\n")
            f.write("-" * 50 + "\n")
            
            for val in param_values:
                key = f"{param_name}_{val}"
                if 'wdbc' in all_results and key in all_results['wdbc']:
                    result = all_results['wdbc'][key]
                    if result is not None:
                        f.write(f"{val:>10} {result['auc_roc']:>14.4f}\n")
                    else:
                        f.write(f"{val:>10} {'N/A':>14}\n")
                else:
                    f.write(f"{val:>10} {'N/A':>14}\n")
            
            f.write("-" * 50 + "\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("OPTIMAL PARAMETER VALUES\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Parameter':<30} {'Default':>12} {'Optimal':>12} {'AUC-ROC':>15}\n")
        f.write("-" * 70 + "\n")
        
        for param_name, param_info in param_configs.items():
            if param_name in optimal_results:
                opt = optimal_results[param_name]
                f.write(f"{param_info['label']:<30} {str(param_info['default']):>12} {str(opt['optimal']):>12} {opt['auc']:>14.4f}\n")
        
        f.write("-" * 70 + "\n")
    
    print(f"\nAll results saved to {summary_path}")
    
    # Plot combined 2x2 sensitivity curves
    plot_combined_sensitivity_curves(
        all_results, 
        param_configs,
        os.path.join(results_dir, 'sensitivity_combined.tiff')
    )
    
    print("\nParameter Sensitivity Analysis Completed!")
    print(f"Generated 1 combined figure (2x2) and 1 optimal values table in {results_dir}/")


if __name__ == "__main__":
    main()