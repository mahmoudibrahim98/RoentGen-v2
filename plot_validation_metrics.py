#!/usr/bin/env python3
"""
Plot validation metrics from validation manifest JSON file.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import re

def load_metrics(json_path):
    """Load metrics from validation manifest JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    steps = []
    metrics_dict = {}
    
    def extract_step_number(key):
        """Extract step number from key, handling both 'checkpoint-XXXX' and 'step_XXXX' formats."""
        if 'checkpoint-' in key:
            return int(key.split('checkpoint-')[1])
        elif '_' in key:
            parts = key.split('_')
            if len(parts) > 1:
                return int(parts[1])
        # Fallback: try to extract any number from the key
        numbers = re.findall(r'\d+', key)
        if numbers:
            return int(numbers[0])
        return 0
    
    for step_key, step_data in sorted(data.items(), key=lambda x: extract_step_number(x[0])):
        step = extract_step_number(step_key)
        
        if 'metrics' in step_data and step_data['metrics'] is not None:
            steps.append(step)
            for metric_name, metric_value in step_data['metrics'].items():
                if metric_name not in metrics_dict:
                    metrics_dict[metric_name] = []
                metrics_dict[metric_name].append(metric_value)
    
    return steps, metrics_dict

def load_real_data_metrics(json_path):
    """Load real data metrics from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'metrics' not in data:
            return None
        
        # Map real data metric names to synthetic metric names (with val/ prefix)
        real_metrics = {}
        metric_mapping = {
            'Atelectasis': 'val/Atelectasis',
            'Cardiomegaly': 'val/Cardiomegaly',
            'Edema': 'val/Edema',
            'Pneumothorax': 'val/Pneumothorax',
            'Effusion': 'val/Effusion',
            'mean_auroc': 'val/mean_auroc',
            'sex_accuracy': 'val/sex_accuracy',
            'race_accuracy': 'val/race_accuracy',
            'age_rmse': 'val/age_rmse',
            'fid': 'val/fid',
            'fid_radimagenet': 'val/fid_radimagenet',
            'biovil_similarity': 'val/biovil_similarity',
            'ms_ssim': 'val/ms_ssim'
        }
        
        for real_key, synthetic_key in metric_mapping.items():
            if real_key in data['metrics']:
                real_metrics[synthetic_key] = data['metrics'][real_key]
        
        return real_metrics
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load real data metrics from {json_path}: {e}")
        return None

def plot_metrics(steps, metrics_dict, output_dir=None, real_data_metrics=None):
    """Plot all metrics over training steps."""
    
    # Group metrics by category
    similarity_metrics = ['val/biovil_similarity', 'val/ms_ssim']  # Removed FID - now in separate plot
    disease_metrics = ['val/Atelectasis', 'val/Cardiomegaly', 'val/Edema', 'val/Pneumothorax', 'val/Effusion', 'val/mean_auroc']
    demographic_metrics = ['val/sex_accuracy', 'val/race_accuracy']  # Removed age_rmse - now in separate plot
    
    # Convert steps to numpy array for easier handling
    steps_array = np.array(steps)
    x_min = min(steps) if steps else 0
    x_max = max(steps) if steps else 10000
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Similarity Metrics
    ax1 = plt.subplot(2, 3, 1)
    for metric in similarity_metrics:
        line = None
        if metric in metrics_dict:
            values = np.array(metrics_dict[metric])
            # Handle NaN values
            valid_mask = ~np.isnan(values)
            if np.any(valid_mask):
                line = ax1.plot(np.array(steps)[valid_mask], values[valid_mask], marker='o', label=metric.replace('val/', ''), linewidth=2, markersize=6)
        # Plot real data metric as dashed line
        if real_data_metrics and metric in real_data_metrics:
            line_color = line[0].get_color() if line else 'gray'
            ax1.axhline(y=real_data_metrics[metric], linestyle='--', linewidth=2, 
                       label=f"{metric.replace('val/', '')} (real)", alpha=0.7, color=line_color)
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Metric Value', fontsize=12)
    ax1.set_title('Similarity Metrics', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # 2. Disease AUROC (individual diseases)
    ax2 = plt.subplot(2, 3, 2)
    disease_individual = ['val/Atelectasis', 'val/Cardiomegaly', 'val/Edema', 'val/Pneumothorax', 'val/Effusion']
    for metric in disease_individual:
        if metric in metrics_dict:
            values = np.array(metrics_dict[metric])
            valid_mask = ~np.isnan(values)
            if np.any(valid_mask):
                line = ax2.plot(np.array(steps)[valid_mask], values[valid_mask], marker='o', label=metric.replace('val/', ''), linewidth=2, markersize=6)
                # Plot real data metric as dashed line
                if real_data_metrics and metric in real_data_metrics:
                    ax2.axhline(y=real_data_metrics[metric], linestyle='--', linewidth=2, 
                               label=f"{metric.replace('val/', '')} (real)", alpha=0.7, color=line[0].get_color())
    if 'val/mean_auroc' in metrics_dict:
        values = np.array(metrics_dict['val/mean_auroc'])
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            ax2.plot(np.array(steps)[valid_mask], values[valid_mask], marker='s', label='Mean AUROC', linewidth=3, markersize=8, linestyle='--', color='black')
            # Plot real data mean AUROC as dashed line
            if real_data_metrics and 'val/mean_auroc' in real_data_metrics:
                ax2.axhline(y=real_data_metrics['val/mean_auroc'], linestyle='--', linewidth=3, 
                           label='Mean AUROC (real)', alpha=0.7, color='gray')
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('AUROC', fontsize=12)
    ax2.set_title('Disease Classification AUROC', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    ax2.set_xlim(left=0)
    ax2.axhline(y=0.5, color='r', linestyle=':', alpha=0.5, label='Random (0.5)')
    
    # 3. Mean AUROC (standalone)
    ax3 = plt.subplot(2, 3, 3)
    if 'val/mean_auroc' in metrics_dict:
        values = np.array(metrics_dict['val/mean_auroc'])
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            ax3.plot(np.array(steps)[valid_mask], values[valid_mask], marker='o', label='Mean AUROC', linewidth=3, markersize=8, color='darkblue')
            ax3.fill_between(np.array(steps)[valid_mask], values[valid_mask], alpha=0.3, color='darkblue')
    # Plot real data mean AUROC as dashed line
    if real_data_metrics and 'val/mean_auroc' in real_data_metrics:
        ax3.axhline(y=real_data_metrics['val/mean_auroc'], linestyle='--', linewidth=3, 
                   label='Mean AUROC (real)', alpha=0.7, color='darkblue')
    ax3.set_xlabel('Training Step', fontsize=12)
    ax3.set_ylabel('Mean AUROC', fontsize=12)
    ax3.set_title('Mean Disease AUROC', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    ax3.set_xlim(left=0)
    ax3.axhline(y=0.5, color='r', linestyle=':', alpha=0.5, label='Random (0.5)')
    ax3.legend(fontsize=10)
    
    # 4. Demographic Metrics (without age_rmse)
    ax4 = plt.subplot(2, 3, 4)
    for metric in demographic_metrics:
        if metric in metrics_dict:
            values = np.array(metrics_dict[metric])
            valid_mask = ~np.isnan(values)
            if np.any(valid_mask):
                label = metric.replace('val/', '').replace('_', ' ').title()
                line = ax4.plot(np.array(steps)[valid_mask], values[valid_mask], marker='o', label=label, linewidth=2, markersize=6)
                # Plot real data metric as dashed line
                if real_data_metrics and metric in real_data_metrics:
                    ax4.axhline(y=real_data_metrics[metric], linestyle='--', linewidth=2, 
                               label=f"{label} (real)", alpha=0.7, color=line[0].get_color())
    ax4.set_xlabel('Training Step', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('Demographic Prediction Metrics', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(left=0)
    ax4.set_ylim([0, 1])
    
    # 5. FID and Age RMSE (lower is better - separate plot with dual y-axes)
    ax5 = plt.subplot(2, 3, 5)
    ax5_twin = ax5.twinx()  # Create second y-axis
    
    # Plot FID on left y-axis
    if 'val/fid' in metrics_dict:
        values = np.array(metrics_dict['val/fid'])
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            ax5.plot(np.array(steps)[valid_mask], values[valid_mask], marker='o', label='FID', 
                    linewidth=3, markersize=8, color='darkred')
            ax5.fill_between(np.array(steps)[valid_mask], values[valid_mask], alpha=0.2, color='darkred')
    # Plot real data FID as dashed line
    if real_data_metrics and 'val/fid' in real_data_metrics:
        ax5.axhline(y=real_data_metrics['val/fid'], linestyle='--', linewidth=3, 
                   label='FID (real)', alpha=0.7, color='darkred')
    
    # Plot FID RadImageNet on left y-axis
    if 'val/fid_radimagenet' in metrics_dict:
        values = np.array(metrics_dict['val/fid_radimagenet'])
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            ax5.plot(np.array(steps)[valid_mask], values[valid_mask], marker='^', label='FID RadImageNet', 
                    linewidth=3, markersize=8, color='crimson')
            ax5.fill_between(np.array(steps)[valid_mask], values[valid_mask], alpha=0.2, color='crimson')
    # Plot real data FID RadImageNet as dashed line
    if real_data_metrics and 'val/fid_radimagenet' in real_data_metrics:
        ax5.axhline(y=real_data_metrics['val/fid_radimagenet'], linestyle='--', linewidth=3, 
                   label='FID RadImageNet (real)', alpha=0.7, color='crimson')
    
    # Plot Age RMSE on right y-axis
    if 'val/age_rmse' in metrics_dict:
        values = np.array(metrics_dict['val/age_rmse'])
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            ax5_twin.plot(np.array(steps)[valid_mask], values[valid_mask], marker='s', label='Age RMSE', 
                         linewidth=3, markersize=8, color='darkblue')
            ax5_twin.fill_between(np.array(steps)[valid_mask], values[valid_mask], alpha=0.2, color='darkblue')
    # Plot real data Age RMSE as dashed line
    if real_data_metrics and 'val/age_rmse' in real_data_metrics:
        ax5_twin.axhline(y=real_data_metrics['val/age_rmse'], linestyle='--', linewidth=3, 
                       label='Age RMSE (real)', alpha=0.7, color='darkblue')
    
    ax5.set_xlabel('Training Step', fontsize=12)
    ax5.set_ylabel('FID Score', fontsize=12, color='darkred')
    ax5_twin.set_ylabel('Age RMSE', fontsize=12, color='darkblue')
    ax5.set_title('FID & Age RMSE (Lower is Better)', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(left=0)
    
    # Combine legends
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='best')
    
    # Color the y-axis labels
    ax5.tick_params(axis='y', labelcolor='darkred')
    ax5_twin.tick_params(axis='y', labelcolor='darkblue')
    
    # 6. All metrics overview (normalized)
    ax6 = plt.subplot(2, 3, 6)
    all_metrics_to_plot = {
        'val/mean_auroc': {'label': 'Mean AUROC', 'color': 'blue'},
        'val/biovil_similarity': {'label': 'BioViL Similarity', 'color': 'green'},
        'val/ms_ssim': {'label': 'MS-SSIM', 'color': 'orange'},
        'val/sex_accuracy': {'label': 'Sex Accuracy', 'color': 'purple'},
        'val/race_accuracy': {'label': 'Race Accuracy', 'color': 'brown'},
    }
    
    for metric, style in all_metrics_to_plot.items():
        if metric in metrics_dict:
            values = np.array(metrics_dict[metric])
            valid_mask = ~np.isnan(values)
            if np.any(valid_mask):
                # Normalize to [0, 1] for comparison
                valid_values = values[valid_mask]
                if len(valid_values) > 0 and np.max(valid_values) > np.min(valid_values):
                    normalized = (valid_values - np.min(valid_values)) / (np.max(valid_values) - np.min(valid_values))
                else:
                    normalized = valid_values
                ax6.plot(np.array(steps)[valid_mask], normalized, marker='o', label=style['label'], 
                        linewidth=2, markersize=5, color=style['color'])
    
    ax6.set_xlabel('Training Step', fontsize=12)
    ax6.set_ylabel('Normalized Metric Value', fontsize=12)
    ax6.set_title('Normalized Metrics Overview', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=9, loc='best')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(left=0)
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        output_path = Path(output_dir) / 'validation_metrics_plot.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.savefig('validation_metrics_plot.png', dpi=300, bbox_inches='tight')
        print("Saved plot to validation_metrics_plot.png")
    
    plt.show()

def print_summary(steps, metrics_dict):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("VALIDATION METRICS SUMMARY")
    print("="*60)
    
    if 'val/mean_auroc' in metrics_dict:
        values = np.array(metrics_dict['val/mean_auroc'])
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            print(f"\nMean AUROC:")
            print(f"  Best: {np.max(valid_values):.4f} at step {steps[np.argmax(valid_values)]}")
            print(f"  Latest: {valid_values[-1]:.4f} at step {steps[-1]}")
            print(f"  Average: {np.mean(valid_values):.4f}")
    
    if 'val/fid' in metrics_dict:
        values = np.array(metrics_dict['val/fid'])
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            print(f"\nFID (lower is better):")
            print(f"  Best: {np.min(valid_values):.4f} at step {steps[np.argmin(valid_values)]}")
            print(f"  Latest: {valid_values[-1]:.4f} at step {steps[-1]}")
            print(f"  Average: {np.mean(valid_values):.4f}")
    
    if 'val/fid_radimagenet' in metrics_dict:
        values = np.array(metrics_dict['val/fid_radimagenet'])
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            print(f"\nFID RadImageNet (lower is better):")
            print(f"  Best: {np.min(valid_values):.4f} at step {steps[np.argmin(valid_values)]}")
            print(f"  Latest: {valid_values[-1]:.4f} at step {steps[-1]}")
            print(f"  Average: {np.mean(valid_values):.4f}")
    
    if 'val/biovil_similarity' in metrics_dict:
        values = np.array(metrics_dict['val/biovil_similarity'])
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            print(f"\nBioViL Similarity:")
            print(f"  Best: {np.max(valid_values):.4f} at step {steps[np.argmax(valid_values)]}")
            print(f"  Latest: {valid_values[-1]:.4f} at step {steps[-1]}")
    
    if 'val/sex_accuracy' in metrics_dict:
        values = np.array(metrics_dict['val/sex_accuracy'])
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            print(f"\nSex Accuracy:")
            print(f"  Best: {np.max(valid_values):.4f} at step {steps[np.argmax(valid_values)]}")
            print(f"  Latest: {valid_values[-1]:.4f} at step {steps[-1]}")
    
    if 'val/race_accuracy' in metrics_dict:
        values = np.array(metrics_dict['val/race_accuracy'])
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            print(f"\nRace Accuracy:")
            print(f"  Best: {np.max(valid_values):.4f} at step {steps[np.argmax(valid_values)]}")
            print(f"  Latest: {valid_values[-1]:.4f} at step {steps[-1]}")
    
    if 'val/age_rmse' in metrics_dict:
        values = np.array(metrics_dict['val/age_rmse'])
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            print(f"\nAge RMSE (lower is better):")
            print(f"  Best: {np.min(valid_values):.4f} at step {steps[np.argmin(valid_values)]}")
            print(f"  Latest: {valid_values[-1]:.4f} at step {steps[-1]}")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Plot validation metrics from manifest JSON file')
    parser.add_argument('--manifest', type=str, default='output/test_run/validation_manifest_correct.json',
                        help='Path to validation manifest JSON file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plot (default: current directory)')
    parser.add_argument('--real_data_metrics', type=str, default=None,
                        help='Path to real data metrics JSON file')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plot (only save)')
    
    args = parser.parse_args()
    
    # Load metrics
    print(f"Loading metrics from {args.manifest}...")
    steps, metrics_dict = load_metrics(args.manifest)
    print(f"Loaded {len(steps)} validation steps")
    
    # Load real data metrics if provided
    real_data_metrics = None
    if args.real_data_metrics:
        print(f"Loading real data metrics from {args.real_data_metrics}...")
        real_data_metrics = load_real_data_metrics(args.real_data_metrics)
        if real_data_metrics:
            print(f"Loaded {len(real_data_metrics)} real data metrics")
    else:
        # Try default path
        default_path = Path(__file__).parent / 'real_data_metrics' / 'real_data_metrics_val_20251112_145453.json'
        if default_path.exists():
            print(f"Loading real data metrics from default path: {default_path}...")
            real_data_metrics = load_real_data_metrics(str(default_path))
            if real_data_metrics:
                print(f"Loaded {len(real_data_metrics)} real data metrics")
    
    # Print summary
    print_summary(steps, metrics_dict)
    
    # Plot metrics
    print("\nGenerating plots...")
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    
    plot_metrics(steps, metrics_dict, output_dir=args.output_dir, real_data_metrics=real_data_metrics)

if __name__ == '__main__':
    main()



