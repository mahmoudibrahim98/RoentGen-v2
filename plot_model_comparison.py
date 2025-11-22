#!/usr/bin/env python3
"""
Compare metrics across different models at a specific checkpoint.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict
import re
import pandas as pd
import glob

def extract_metrics_at_step(manifest_path, step):
    """Extract metrics at a specific step from manifest."""
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    
    step_key = f"checkpoint-{step}"
    if step_key not in data:
        # Try alternative format
        step_key = f"step_{step}"
        if step_key not in data:
            return None
    
    step_data = data[step_key]
    if step_data.get('metrics') is None:
        return None
    
    return step_data['metrics']

def compute_subgroup_sizes(validation_data_path):
    """Compute subgroup sizes from validation data CSV or directory."""
    subgroup_sizes = defaultdict(int)
    intersectional_sizes = defaultdict(int)
    
    try:
        # Try to load CSV file
        if Path(validation_data_path).is_file() and validation_data_path.endswith('.csv'):
            df = pd.read_csv(validation_data_path)
        elif Path(validation_data_path).is_dir():
            # Look for CSV files in directory
            csv_files = glob.glob(str(Path(validation_data_path) / "*.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
            else:
                # Try to load from tar files metadata (WebDataset)
                return subgroup_sizes, intersectional_sizes
        else:
            return subgroup_sizes, intersectional_sizes
        
        # Map column names (try common variations)
        age_col = None
        sex_col = None
        race_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'age' in col_lower and age_col is None:
                age_col = col
            elif ('sex' in col_lower or 'gender' in col_lower) and sex_col is None:
                sex_col = col
            elif ('race' in col_lower or 'ethnicity' in col_lower) and race_col is None:
                race_col = col
        
        if age_col is None or sex_col is None or race_col is None:
            return subgroup_sizes, intersectional_sizes
        
        # Compute subgroup sizes
        for _, row in df.iterrows():
            age = row[age_col]
            sex = row[sex_col]
            race = row[race_col]
            
            # Convert age to age group
            try:
                age_val = float(age)
                if age_val < 40:
                    age_group = "18-40"
                elif age_val < 60:
                    age_group = "40-60"
                elif age_val < 80:
                    age_group = "60-80"
                else:
                    age_group = "80+"
            except:
                age_group = str(age)
            
            # Normalize sex
            sex_str = str(sex).strip().title()
            if 'female' in sex_str.lower() or sex_str.lower() == 'f':
                sex_str = "Female"
            elif 'male' in sex_str.lower() or sex_str.lower() == 'm':
                sex_str = "Male"
            
            # Normalize race
            race_str = str(race).strip().title()
            
            # Count subgroups
            subgroup_sizes[f"sex_{sex_str}"] += 1
            subgroup_sizes[f"race_{race_str}"] += 1
            subgroup_sizes[f"age_group_{age_group}"] += 1
            
            # Count intersectional
            intersectional_key = f"{age_group}_{sex_str}_{race_str}"
            intersectional_sizes[intersectional_key] += 1
        
    except Exception as e:
        print(f"Warning: Could not compute subgroup sizes from {validation_data_path}: {e}")
    
    return dict(subgroup_sizes), dict(intersectional_sizes)

def categorize_metrics(metrics_dict):
    """Categorize metrics into main, subgroup, and intersectional."""
    main_metrics = {}
    subgroup_metrics = defaultdict(dict)
    intersectional_metrics = defaultdict(dict)
    
    for metric_name, metric_value in metrics_dict.items():
        if metric_value is None or (isinstance(metric_value, float) and np.isnan(metric_value)):
            continue
            
        # Main metrics (no subgroup or intersectional)
        if 'subgroup' not in metric_name and 'intersectional' not in metric_name:
            main_metrics[metric_name] = metric_value
        
        # Subgroup metrics
        elif 'subgroup' in metric_name:
            # Extract subgroup type and value
            # Format: val/metric_subgroup_type_value
            parts = metric_name.split('_subgroup_')
            if len(parts) == 2:
                base_metric = parts[0]
                subgroup_part = parts[1]
                # Extract subgroup type (sex, race, age_group) and value
                if '_' in subgroup_part:
                    subgroup_type = subgroup_part.split('_')[0]
                    subgroup_value = '_'.join(subgroup_part.split('_')[1:])
                    subgroup_metrics[base_metric][f"{subgroup_type}_{subgroup_value}"] = metric_value
        
        # Intersectional metrics
        elif 'intersectional' in metric_name:
            # Format: val/metric_intersectional_age_sex_race
            parts = metric_name.split('_intersectional_')
            if len(parts) == 2:
                base_metric = parts[0]
                intersectional_key = parts[1]
                intersectional_metrics[base_metric][intersectional_key] = metric_value
    
    return main_metrics, dict(subgroup_metrics), dict(intersectional_metrics)

def plot_main_metrics_comparison(models_data, output_dir, step=None):
    """Plot comparison of main metrics across models, split into themed figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step_suffix = f"_step_{step}" if step is not None else ""
    step_label = f" (Step {step})" if step is not None else ""
    # Extract main metrics
    main_metrics_all = {}
    for model_name, metrics in models_data.items():
        main_metrics, _, _ = categorize_metrics(metrics)
        main_metrics_all[model_name] = main_metrics
    
    # Get all unique metric names
    all_metric_names = set()
    for metrics in main_metrics_all.values():
        all_metric_names.update(metrics.keys())
    
    # Group metrics by category
    model_names = list(models_data.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6C5B7B', '#20BF55', '#F6511D', '#FFB300', '#008080']
    short_names = {
        'roentgen_v2_baseline': 'Baseline',
        '2_hcn_without_uncertainty': 'HCN (no uncertainty)',
        '0_full_hcn': 'Full HCN',
        '0a_full_hcn_strong': 'Full HCN (strong)',
        'baseline': 'Baseline',
        'wrong_hcn': 'HCN (orig v1)',
        '0b_full_hcn_strongest': 'Full HCN (strongest)',
    }
    
    groups = [
        (
            'model_comparison_main_metrics_quality',
            ['val/fid', 'val/fid_radimagenet', 'val/biovil_similarity', 'val/ms_ssim', 'val/mean_auroc'],
            f'Quality & Similarity Metrics{step_label}'
        ),
        (
            'model_comparison_main_metrics_auc',
            ['val/Atelectasis', 'val/Cardiomegaly', 'val/Edema', 'val/Pneumothorax', 'val/Effusion'],
            f'Disease AUROC Metrics{step_label}'
        ),
        (
            'model_comparison_main_metrics_demographics',
            ['val/sex_accuracy', 'val/race_accuracy', 'val/age_rmse'],
            f'Demographic Prediction Metrics{step_label}'
        )
    ]
    
    def _plot_group(metric_names, title, output_name):
        existing_metrics = [m for m in metric_names if m in all_metric_names]
        if not existing_metrics:
            print(f"No metrics found for {output_name}. Skipping.")
            return False
        
        n_cols = min(3, len(existing_metrics))
        n_rows = (len(existing_metrics) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axes = np.array(axes).reshape(-1)
        
        for idx, metric_name in enumerate(existing_metrics):
            ax = axes[idx]
            values = []
            for model_name in model_names:
                value = main_metrics_all[model_name].get(metric_name)
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    values.append(value)
                else:
                    values.append(None)
            
            x_pos = np.arange(len(model_names))
            bar_colors = colors[:len(model_names)]
            if len(bar_colors) < len(model_names):
                bar_colors = None
            bars = ax.bar(x_pos, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=2)
            
            for bar, val in zip(bars, values):
                if val is None:
                    continue
                height = bar.get_height()
                if 'fid' in metric_name.lower():
                    label = f'{val:.1f}'
                elif 'rmse' in metric_name.lower():
                    label = f'{val:.2f}'
                else:
                    label = f'{val:.3f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        label,
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            valid_values = [(i, v) for i, v in enumerate(values) if v is not None]
            if valid_values:
                if 'fid' in metric_name.lower() or 'rmse' in metric_name.lower():
                    best_idx, _ = min(valid_values, key=lambda x: x[1])
                else:
                    best_idx, _ = max(valid_values, key=lambda x: x[1])
                bars[best_idx].set_edgecolor('gold')
                bars[best_idx].set_linewidth(3)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                [short_names.get(name, name.replace('_', ' ').title()) for name in model_names],
                rotation=15, ha='right', fontsize=10
            )
            ax.set_ylabel('Value', fontsize=10)
            metric_display = metric_name.replace('val/', '').replace('_', ' ').title()
            ax.set_title(metric_display, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        for idx in range(len(existing_metrics), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        output_path = output_dir / f'{output_name}{step_suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {title} to {output_path}")
        plt.close(fig)
        return True
    
    plotted = False
    for output_name, metric_names, title in groups:
        plotted = _plot_group(metric_names, title, output_name) or plotted
    
    if not plotted:
        print("No main metrics found to plot")

def plot_subgroup_metrics_comparison(models_data, output_path, subgroup_sizes=None):
    """Plot comparison of subgroup metrics across models."""
    # Extract subgroup metrics
    subgroup_metrics_all = {}
    for model_name, metrics in models_data.items():
        _, subgroup_metrics, _ = categorize_metrics(metrics)
        subgroup_metrics_all[model_name] = subgroup_metrics
    
    # Get all base metrics that have subgroups
    all_base_metrics = set()
    for subgroup_dict in subgroup_metrics_all.values():
        all_base_metrics.update(subgroup_dict.keys())
    
    # Focus on key metrics
    key_base_metrics = ['val/fid', 'val/fid_radimagenet', 'val/ms_ssim']
    key_base_metrics = [m for m in key_base_metrics if m in all_base_metrics]
    
    if not key_base_metrics:
        print("No subgroup metrics found to plot")
        return
    
    model_names = list(models_data.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6C5B7B', '#20BF55', '#F6511D', '#FFB300', '#008080']
    short_names = {
        'roentgen_v2_baseline': 'Baseline',
        '2_hcn_without_uncertainty': 'HCN (no uncertainty)',
        '0_full_hcn': 'Full HCN',
        '0a_full_hcn_strong': 'Full HCN (strong)',
        '0b_full_hcn_strongest': 'Full HCN (strongest)',
        'wrong_hcn': 'HCN (orig v1)',
        'baseline': 'Baseline',
    }
    
    # Create separate plots for each metric type
    fig = plt.figure(figsize=(24, 10))
    
    for base_idx, base_metric in enumerate(key_base_metrics):
        # Get all subgroups for this metric
        all_subgroups = set()
        for model_name in model_names:
            if base_metric in subgroup_metrics_all[model_name]:
                all_subgroups.update(subgroup_metrics_all[model_name][base_metric].keys())
        
        all_subgroups = sorted(all_subgroups)
        
        if not all_subgroups:
            continue
        
        # Group by subgroup type
        sex_subgroups = sorted([s for s in all_subgroups if s.startswith('sex_')])
        race_subgroups = sorted([s for s in all_subgroups if s.startswith('race_')])
        age_subgroups = sorted([s for s in all_subgroups if s.startswith('age_group_')])
        
        # Create subplot for each subgroup type
        for subplot_idx, (subgroup_type, subgroups) in enumerate([
            ('Sex', sex_subgroups),
            ('Race', race_subgroups),
            ('Age Group', age_subgroups)
        ]):
            if not subgroups:
                continue
            
            ax = plt.subplot(len(key_base_metrics), 3, base_idx * 3 + subplot_idx + 1)
            
            x = np.arange(len(subgroups))
            width = 0.25
            
            for model_idx, model_name in enumerate(model_names):
                values = []
                for subgroup in subgroups:
                    value = subgroup_metrics_all[model_name].get(base_metric, {}).get(subgroup)
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        values.append(0)
                    else:
                        values.append(value)
                
                offset = (model_idx - len(model_names)/2 + 0.5) * width
                bars = ax.bar(x + offset, values, width, label=short_names.get(model_name, model_name), 
                             color=colors[model_idx], alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add subgroup sizes as text annotations
            if subgroup_sizes:
                for i, subgroup in enumerate(subgroups):
                    size_key = subgroup.replace('sex_', '').replace('race_', '').replace('age_group_', '')
                    # Try to find matching size
                    size = subgroup_sizes.get(subgroup, 0)
                    if size > 0:
                        ax.text(i, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
                               f'n={size}', ha='center', va='top', fontsize=8, style='italic', color='gray')
            
            ax.set_xlabel(f'{subgroup_type} Subgroup', fontsize=11)
            ax.set_ylabel('Metric Value', fontsize=11)
            metric_display = base_metric.replace('val/', '').replace('_', ' ').title()
            ax.set_title(f'{metric_display} - {subgroup_type}', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([s.replace('sex_', '').replace('race_', '').replace('age_group_', '') 
                                for s in subgroups], rotation=45, ha='right', fontsize=9)
            if subplot_idx == 0:
                ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Subgroup Metrics Comparison Across Models (Step 12500)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved subgroup metrics comparison to {output_path}")
    plt.close()

def plot_intersectional_metrics_comparison(models_data, output_path, intersectional_sizes=None):
    """Plot comparison of intersectional metrics across models."""
    # Extract intersectional metrics
    intersectional_metrics_all = {}
    for model_name, metrics in models_data.items():
        _, _, intersectional_metrics = categorize_metrics(metrics)
        intersectional_metrics_all[model_name] = intersectional_metrics
    
    # Get all base metrics that have intersectional metrics
    all_base_metrics = set()
    for intersectional_dict in intersectional_metrics_all.values():
        all_base_metrics.update(intersectional_dict.keys())
    
    # Focus on key metrics
    key_base_metrics = ['val/fid', 'val/fid_radimagenet', 'val/ms_ssim']
    key_base_metrics = [m for m in key_base_metrics if m in all_base_metrics]
    
    if not key_base_metrics:
        print("No intersectional metrics found to plot")
        return
    
    model_names = list(models_data.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6C5B7B', '#20BF55', '#F6511D', '#FFB300', '#008080']
    short_names = {
        'roentgen_v2_baseline': 'Baseline',
        '2_hcn_without_uncertainty': 'HCN (no uncertainty)',
        '0_full_hcn': 'Full HCN',
        '0a_full_hcn_strong': 'Full HCN (strong)',
        '0b_full_hcn_strongest': 'Full HCN (strongest)',
        'wrong_hcn': 'HCN (orig v1)',
        'baseline': 'Baseline',
    }
    
    # Create separate plots for each metric, grouped by age group
    fig = plt.figure(figsize=(24, 12))
    
    for base_idx, base_metric in enumerate(key_base_metrics):
        # Get all intersectional groups for this metric
        all_intersectional = set()
        for model_name in model_names:
            if base_metric in intersectional_metrics_all[model_name]:
                all_intersectional.update(intersectional_metrics_all[model_name][base_metric].keys())
        
        all_intersectional = sorted(all_intersectional)
        
        if not all_intersectional:
            continue
        
        # Group by age group
        age_groups = ['18-40', '40-60', '60-80', '80+']
        
        for age_idx, age_group in enumerate(age_groups):
            # Filter intersectional groups for this age group
            age_intersectional = [i for i in all_intersectional if i.startswith(age_group)]
            if not age_intersectional:
                continue
            
            ax = plt.subplot(len(key_base_metrics), len(age_groups), base_idx * len(age_groups) + age_idx + 1)
            
            x = np.arange(len(age_intersectional))
            width = 0.25
            
            for model_idx, model_name in enumerate(model_names):
                values = []
                for intersectional_key in age_intersectional:
                    value = intersectional_metrics_all[model_name].get(base_metric, {}).get(intersectional_key)
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        values.append(0)
                    else:
                        values.append(value)
                
                offset = (model_idx - len(model_names)/2 + 0.5) * width
                bars = ax.bar(x + offset, values, width, label=short_names.get(model_name, model_name), 
                             color=colors[model_idx], alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add intersectional sizes as text annotations
            if intersectional_sizes:
                for i, intersectional_key in enumerate(age_intersectional):
                    size = intersectional_sizes.get(intersectional_key, 0)
                    if size > 0:
                        ax.text(i, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
                               f'n={size}', ha='center', va='top', fontsize=7, style='italic', color='gray')
            
            ax.set_xlabel('Sex_Race', fontsize=10)
            ax.set_ylabel('Metric Value', fontsize=10)
            metric_display = base_metric.replace('val/', '').replace('_', ' ').title()
            ax.set_title(f'{metric_display} - Age {age_group}', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([s.replace(f'{age_group}_', '') for s in age_intersectional], 
                             rotation=45, ha='right', fontsize=8)
            if base_idx == 0 and age_idx == 0:
                ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Intersectional Metrics Comparison Across Models (Step 12500)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved intersectional metrics comparison to {output_path}")
    plt.close()

def create_summary_table(models_data, output_path):
    """Create a summary table of key metrics."""
    # Extract main metrics
    main_metrics_all = {}
    for model_name, metrics in models_data.items():
        main_metrics, _, _ = categorize_metrics(metrics)
        main_metrics_all[model_name] = main_metrics
    
    short_names = {
        'roentgen_v2_baseline': 'Baseline',
        '2_hcn_without_uncertainty': 'HCN (no uncertainty)',
        '0_full_hcn': 'Full HCN'
    }
    
    # Key metrics to include, grouped by category
    metric_groups = [
        ('Fidelity Metrics', ['val/fid', 'val/fid_radimagenet']),
        ('Similarity Metrics', ['val/biovil_similarity', 'val/ms_ssim']),
        ('Disease Classification', ['val/mean_auroc', 'val/Atelectasis', 'val/Cardiomegaly', 'val/Edema', 'val/Pneumothorax', 'val/Effusion']),
        ('Demographic Prediction', ['val/sex_accuracy', 'val/race_accuracy', 'val/age_rmse'])
    ]
    
    # Create table
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    table_data = []
    headers = ['Metric'] + [short_names.get(name, name.replace('_', ' ').title()) for name in models_data.keys()]
    
    for group_name, metrics in metric_groups:
        # Add group header
        table_data.append([f'\n{group_name}', '', '', ''])
        
        for metric in metrics:
            # Check if metric exists in any model
            exists = any(main_metrics_all[model_name].get(metric) is not None and 
                        not (isinstance(main_metrics_all[model_name].get(metric), float) and 
                             np.isnan(main_metrics_all[model_name].get(metric)))
                        for model_name in models_data.keys())
            
            if not exists:
                continue
            
            row = [f'  {metric.replace("val/", "").replace("_", " ").title()}']
            for model_name in models_data.keys():
                value = main_metrics_all[model_name].get(metric)
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    # Format based on metric type
                    if 'fid' in metric.lower():
                        row.append(f'{value:.1f}')
                    elif 'rmse' in metric.lower():
                        row.append(f'{value:.2f}')
                    else:
                        row.append(f'{value:.3f}')
                else:
                    row.append('N/A')
            table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style group headers
    row_idx = 1
    for group_name, metrics in metric_groups:
        table[(row_idx, 0)].set_facecolor('#E8F4F8')
        table[(row_idx, 0)].set_text_props(weight='bold', style='italic')
        for i in range(1, len(headers)):
            table[(row_idx, i)].set_facecolor('#E8F4F8')
        row_idx += len([m for m in metrics if any(main_metrics_all[n].get(m) is not None and 
                                                   not (isinstance(main_metrics_all[n].get(m), float) and 
                                                        np.isnan(main_metrics_all[n].get(m)))
                                                   for n in models_data.keys())]) + 1
    
    plt.title('Model Comparison Summary - Step 12500', fontsize=18, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary table to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare metrics across models at a specific step')
    parser.add_argument('--step', type=int, default=12500, help='Step number to compare')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for plots')
    parser.add_argument('--validation_data', type=str, default=None, 
                       help='Path to validation data CSV file or directory containing CSV files')
    
    args = parser.parse_args()
    
    # Model paths
    # base_dir = Path(__file__).parent / 'output'
    # models = {
    #     'roentgen_v2_baseline': base_dir / 'roentgen_v2_baseline' / 'validation_manifest.json',
    #     '2_hcn_without_uncertainty': base_dir / '2_hcn_without_uncertainty' / 'validation_manifest.json',
    #     '0_full_hcn': base_dir / '0_full_hcn' / 'validation_manifest.json'
    # }
    base_dir = Path(__file__).parent / 'outputs/output'
    base_dir_v2 = Path(__file__).parent / 'outputs/output_v2'
    base_dir_v4 = Path(__file__).parent / 'outputs/output_v4'
    base_dir_v3 = Path(__file__).parent / 'outputs/output_v3'
    models = {
        'baseline': base_dir / '1_baseline' / 'validation_manifest.json',
        'wrong_hcn': base_dir / '0_full_hcn' / 'validation_manifest.json',
        '0_full_hcn': base_dir_v2 / '0_full_hcn' / 'validation_manifest.json',
        '0a_full_hcn_strong': base_dir_v2 / '0a_full_hcn_strong' / 'validation_manifest.json',
        '0b_full_hcn_strongest': base_dir_v2 / '0b_full_hcn_strongest' / 'validation_manifest.json',
        
    }
    # Try to compute subgroup sizes from validation data
    validation_data_path = args.validation_data
    if validation_data_path is None:
        # Try common validation data paths
        try:
            possible_paths = [
                Path(__file__).parent / 'real_data' / 'val_data',
                Path(__file__).parent / 'real_data' / 'validation_data',
            ]
            for path in possible_paths:
                if path.exists():
                    validation_data_path = str(path)
                    break
        except:
            pass
    
    subgroup_sizes = None
    intersectional_sizes = None
    if validation_data_path:
        print(f"Computing subgroup sizes from {validation_data_path}...")
        subgroup_sizes, intersectional_sizes = compute_subgroup_sizes(validation_data_path)
        if subgroup_sizes:
            print(f"Computed sizes for {len(subgroup_sizes)} subgroups and {len(intersectional_sizes)} intersectional groups")
        else:
            print("Warning: Could not compute subgroup sizes. Plots will be generated without sample sizes.")
    else:
        print("Note: No validation data path provided. Plots will be generated without sample sizes.")
    
    # Load metrics for each model
    models_data = {}
    for model_name, manifest_path in models.items():
        if not manifest_path.exists():
            print(f"Warning: Manifest not found for {model_name}: {manifest_path}")
            continue
        
        metrics = extract_metrics_at_step(str(manifest_path), args.step)
        if metrics is None:
            print(f"Warning: No metrics found for {model_name} at step {args.step}")
            continue
        
        models_data[model_name] = metrics
        print(f"Loaded {len(metrics)} metrics for {model_name}")
    
    if not models_data:
        print("Error: No model data loaded")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating comparison plots...")
    
    # Main metrics comparison
    plot_main_metrics_comparison(
        models_data, 
        output_dir,
        step=args.step
    )
    
    # Subgroup metrics comparison
    plot_subgroup_metrics_comparison(
        models_data,
        output_dir / f'model_comparison_subgroup_metrics_step_{args.step}.png',
        subgroup_sizes=subgroup_sizes
    )
    
    # Intersectional metrics comparison
    plot_intersectional_metrics_comparison(
        models_data,
        output_dir / f'model_comparison_intersectional_metrics_step_{args.step}.png',
        intersectional_sizes=intersectional_sizes
    )
    
    # Summary table
    create_summary_table(
        models_data,
        output_dir / f'model_comparison_summary_step_{args.step}.png'
    )
    
    print(f"\nAll comparison plots saved to {output_dir}")

if __name__ == '__main__':
    main()

