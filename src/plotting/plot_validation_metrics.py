#!/usr/bin/env python3
"""
Plot validation metrics from validation manifest JSON file.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def load_metrics(json_path):
    """Load metrics from validation manifest JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    steps = []
    metrics_dict = {}
    
    for step_key, step_data in sorted(data.items(), key=lambda x: int(x[0].split('_')[1])):
        step = int(step_key.split('_')[1])
        
        if 'metrics' in step_data and step_data['metrics'] is not None:
            steps.append(step)
            for metric_name, metric_value in step_data['metrics'].items():
                if metric_name not in metrics_dict:
                    metrics_dict[metric_name] = []
                metrics_dict[metric_name].append(metric_value)
    
    return steps, metrics_dict

def load_real_data_metrics(json_path):
    """Load real data metrics from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'metrics' not in data:
        return None
    
    # Map real data metric names to validation metric names
    # Real data uses names like "Atelectasis", validation uses "val/Atelectasis"
    metrics = {}
    for metric_name, metric_value in data['metrics'].items():
        # Convert to validation format
        if metric_name in ['Atelectasis', 'Cardiomegaly', 'Edema', 'Pneumothorax', 'Effusion']:
            metrics[f'val/{metric_name}'] = metric_value
        elif metric_name == 'mean_auroc':
            metrics['val/mean_auroc'] = metric_value
        elif metric_name == 'sex_accuracy':
            metrics['val/sex_accuracy'] = metric_value
        elif metric_name == 'race_accuracy':
            metrics['val/race_accuracy'] = metric_value
        elif metric_name == 'age_rmse':
            metrics['val/age_rmse'] = metric_value
        elif metric_name == 'fid':
            metrics['val/fid'] = metric_value
        elif metric_name == 'fid_radimagenet':
            metrics['val/fid_radimagenet'] = metric_value
        elif metric_name == 'biovil_similarity':
            metrics['val/biovil_similarity'] = metric_value
        elif metric_name == 'ms_ssim':
            metrics['val/ms_ssim'] = metric_value
    
    return metrics

def plot_metrics(steps, metrics_dict, output_dir=None, no_show=False, real_data_metrics=None):
    """Plot all metrics over training steps.
    
    Args:
        steps: List of training steps
        metrics_dict: Dictionary of metric names to lists of values
        output_dir: Output directory for plots
        no_show: Whether to suppress showing plots
        real_data_metrics: Dictionary of real data metrics to plot as reference lines
    """
    
    # Group metrics by category (excluding FID and age_rmse - they get separate plots)
    similarity_metrics = ['val/biovil_similarity', 'val/ms_ssim']  # Removed FID
    disease_metrics = ['val/Atelectasis', 'val/Cardiomegaly', 'val/Edema', 'val/Pneumothorax', 'val/Effusion', 'val/mean_auroc']
    demographic_metrics = ['val/sex_accuracy', 'val/race_accuracy']  # Removed age_rmse
    
    # Convert steps to numpy array for easier handling
    steps_array = np.array(steps)
    
    # ============================================================
    # MAIN FIGURE: 2x3 grid with 6 plots
    # ============================================================
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Similarity Metrics (without FID)
    ax1 = plt.subplot(2, 3, 1)
    # Define colors for similarity metrics
    similarity_colors = {
        'val/biovil_similarity': 'green',
        'val/ms_ssim': 'orange'
    }
    for metric in similarity_metrics:
        color = similarity_colors.get(metric, None)
        if metric in metrics_dict:
            values = np.array(metrics_dict[metric])
            # Handle NaN values
            valid_mask = ~np.isnan(values)
            if np.any(valid_mask):
                ax1.plot(np.array(steps)[valid_mask], values[valid_mask], marker='o', 
                        label=metric.replace('val/', ''), linewidth=2, markersize=6, color=color)
        # Add real data reference line with matching color
        if real_data_metrics and metric in real_data_metrics:
            real_value = real_data_metrics[metric]
            if not np.isnan(real_value) and color:
                ax1.axhline(y=real_value, color=color, linestyle='--', linewidth=2, alpha=0.7, 
                           label=f'Real Data ({metric.replace("val/", "")})')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Metric Value', fontsize=12)
    ax1.set_title('Similarity Metrics (BioViL, MS-SSIM)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # 2. Disease AUROC (individual diseases)
    ax2 = plt.subplot(2, 3, 2)
    disease_individual = ['val/Atelectasis', 'val/Cardiomegaly', 'val/Edema', 'val/Pneumothorax', 'val/Effusion']
    # Define colors for diseases
    disease_colors = {
        'val/Atelectasis': 'blue',
        'val/Cardiomegaly': 'green',
        'val/Edema': 'orange',
        'val/Pneumothorax': 'red',
        'val/Effusion': 'purple'
    }
    for metric in disease_individual:
        color = disease_colors.get(metric, None)
        if metric in metrics_dict:
            values = np.array(metrics_dict[metric])
            valid_mask = ~np.isnan(values)
            if np.any(valid_mask):
                ax2.plot(np.array(steps)[valid_mask], values[valid_mask], marker='o', 
                        label=metric.replace('val/', ''), linewidth=2, markersize=6, color=color)
        # Add real data reference line with matching color
        if real_data_metrics and metric in real_data_metrics:
            real_value = real_data_metrics[metric]
            if not np.isnan(real_value) and color:
                ax2.axhline(y=real_value, color=color, linestyle='--', linewidth=2, alpha=0.7)
    if 'val/mean_auroc' in metrics_dict:
        values = np.array(metrics_dict['val/mean_auroc'])
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            ax2.plot(np.array(steps)[valid_mask], values[valid_mask], marker='s', label='Mean AUROC', 
                    linewidth=3, markersize=8, linestyle='--', color='black')
    # Add real data mean AUROC reference line (black dashed to match)
    if real_data_metrics and 'val/mean_auroc' in real_data_metrics:
        real_value = real_data_metrics['val/mean_auroc']
        if not np.isnan(real_value):
            ax2.axhline(y=real_value, color='black', linestyle='--', linewidth=3, alpha=0.7, label='Real Data (Mean AUROC)')
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('AUROC', fontsize=12)
    ax2.set_title('Disease Classification AUROC', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    ax2.set_xlim(left=0)
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # 3. Mean AUROC (standalone)
    ax3 = plt.subplot(2, 3, 3)
    if 'val/mean_auroc' in metrics_dict:
        values = np.array(metrics_dict['val/mean_auroc'])
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            ax3.plot(np.array(steps)[valid_mask], values[valid_mask], marker='o', label='Mean AUROC', linewidth=3, markersize=8, color='darkblue')
            ax3.fill_between(np.array(steps)[valid_mask], values[valid_mask], alpha=0.3, color='darkblue')
    # Add real data reference line (darkblue to match Mean AUROC)
    if real_data_metrics and 'val/mean_auroc' in real_data_metrics:
        real_value = real_data_metrics['val/mean_auroc']
        if not np.isnan(real_value):
            ax3.axhline(y=real_value, color='darkblue', linestyle='--', linewidth=3, alpha=0.7, label=f'Real Data ({real_value:.4f})')
    ax3.set_xlabel('Training Step', fontsize=12)
    ax3.set_ylabel('Mean AUROC', fontsize=12)
    ax3.set_title('Mean Disease AUROC', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    ax3.set_xlim(left=0)
    ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax3.legend(fontsize=10)
    
    # 4. Demographic Metrics (without Age RMSE)
    ax4 = plt.subplot(2, 3, 4)
    # Define colors for demographic metrics
    demographic_colors = {
        'val/sex_accuracy': 'purple',
        'val/race_accuracy': 'brown'
    }
    for metric in demographic_metrics:
        color = demographic_colors.get(metric, None)
        if metric in metrics_dict:
            values = np.array(metrics_dict[metric])
            valid_mask = ~np.isnan(values)
            if np.any(valid_mask):
                label = metric.replace('val/', '').replace('_', ' ').title()
                ax4.plot(np.array(steps)[valid_mask], values[valid_mask], marker='o', 
                        label=label, linewidth=2, markersize=6, color=color)
        # Add real data reference line with matching color
        if real_data_metrics and metric in real_data_metrics:
            real_value = real_data_metrics[metric]
            if not np.isnan(real_value) and color:
                label_name = metric.replace('val/', '').replace('_', ' ').title()
                ax4.axhline(y=real_value, color=color, linestyle='--', linewidth=2, alpha=0.7, 
                           label=f'Real Data ({label_name})')
    ax4.set_xlabel('Training Step', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('Demographic Prediction (Sex, Race)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    ax4.set_xlim(left=0)
    
    # 5. BioViL Similarity (standalone)
    ax5 = plt.subplot(2, 3, 5)
    if 'val/biovil_similarity' in metrics_dict:
        values = np.array(metrics_dict['val/biovil_similarity'])
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            ax5.plot(np.array(steps)[valid_mask], values[valid_mask], marker='o', label='BioViL Similarity', linewidth=3, markersize=8, color='darkgreen')
            ax5.fill_between(np.array(steps)[valid_mask], values[valid_mask], alpha=0.3, color='darkgreen')
    # Add real data reference line (green to match BioViL)
    if real_data_metrics and 'val/biovil_similarity' in real_data_metrics:
        real_value = real_data_metrics['val/biovil_similarity']
        if not np.isnan(real_value):
            ax5.axhline(y=real_value, color='darkgreen', linestyle='--', linewidth=3, alpha=0.7, label=f'Real Data ({real_value:.4f})')
    ax5.set_xlabel('Training Step', fontsize=12)
    ax5.set_ylabel('BioViL Similarity', fontsize=12)
    ax5.set_title('BioViL Cosine Similarity', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])
    ax5.set_xlim(left=0)
    ax5.legend(fontsize=10)
    
    # 6. All metrics overview (normalized, excluding FID and age_rmse)
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
    
    # Save main figure
    if output_dir:
        output_path = Path(output_dir) / 'validation_metrics_plot.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved main plot to {output_path}")
    else:
        plt.savefig('validation_metrics_plot.png', dpi=300, bbox_inches='tight')
        print("Saved main plot to validation_metrics_plot.png")
    
    plt.close()
    
    # ============================================================
    # SEPARATE FIGURE 1: FID (Fidelity)
    # ============================================================
    fig_fid = plt.figure(figsize=(10, 6))
    ax_fid = fig_fid.add_subplot(111)
    
    if 'val/fid' in metrics_dict:
        values = np.array(metrics_dict['val/fid'])
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            ax_fid.plot(np.array(steps)[valid_mask], values[valid_mask], marker='o', label='FID', linewidth=3, markersize=8, color='darkred')
            ax_fid.fill_between(np.array(steps)[valid_mask], values[valid_mask], alpha=0.3, color='darkred')
            
            # Mark best point
            best_idx = np.argmin(values[valid_mask])
            best_step = np.array(steps)[valid_mask][best_idx]
            best_value = values[valid_mask][best_idx]
            ax_fid.plot(best_step, best_value, marker='*', markersize=20, color='gold', label=f'Best: {best_value:.2f} at step {best_step}')
    
    # Add real data reference line (darkred to match FID)
    if real_data_metrics and 'val/fid' in real_data_metrics:
        real_value = real_data_metrics['val/fid']
        if not np.isnan(real_value):
            ax_fid.axhline(y=real_value, color='darkred', linestyle='--', linewidth=3, alpha=0.7, label=f'Real Data ({real_value:.2f})')
    
    # Plot FID RadImageNet
    if 'val/fid_radimagenet' in metrics_dict:
        values = np.array(metrics_dict['val/fid_radimagenet'])
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            ax_fid.plot(np.array(steps)[valid_mask], values[valid_mask], marker='^', label='FID RadImageNet', linewidth=3, markersize=8, color='crimson')
            ax_fid.fill_between(np.array(steps)[valid_mask], values[valid_mask], alpha=0.3, color='crimson')
            
            # Mark best point
            best_idx = np.argmin(values[valid_mask])
            best_step = np.array(steps)[valid_mask][best_idx]
            best_value = values[valid_mask][best_idx]
            ax_fid.plot(best_step, best_value, marker='*', markersize=20, color='gold', label=f'Best RadImageNet: {best_value:.2f} at step {best_step}')
    
    # Add real data reference line for FID RadImageNet
    if real_data_metrics and 'val/fid_radimagenet' in real_data_metrics:
        real_value = real_data_metrics['val/fid_radimagenet']
        if not np.isnan(real_value):
            ax_fid.axhline(y=real_value, color='crimson', linestyle='--', linewidth=3, alpha=0.7, label=f'Real Data RadImageNet ({real_value:.2f})')
    
    ax_fid.set_xlabel('Training Step', fontsize=14)
    ax_fid.set_ylabel('FID Score', fontsize=14)
    ax_fid.set_title('Fréchet Inception Distance (Fidelity) - Lower is Better', fontsize=16, fontweight='bold')
    ax_fid.grid(True, alpha=0.3)
    ax_fid.set_xlim(left=0)
    ax_fid.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save FID figure
    if output_dir:
        output_path_fid = Path(output_dir) / 'validation_metrics_fid.png'
        plt.savefig(output_path_fid, dpi=300, bbox_inches='tight')
        print(f"Saved FID plot to {output_path_fid}")
    else:
        plt.savefig('validation_metrics_fid.png', dpi=300, bbox_inches='tight')
        print("Saved FID plot to validation_metrics_fid.png")
    
    plt.close()
    
    # ============================================================
    # SEPARATE FIGURE 2: Age RMSE
    # ============================================================
    fig_age = plt.figure(figsize=(10, 6))
    ax_age = fig_age.add_subplot(111)
    
    if 'val/age_rmse' in metrics_dict:
        values = np.array(metrics_dict['val/age_rmse'])
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            ax_age.plot(np.array(steps)[valid_mask], values[valid_mask], marker='o', label='Age RMSE', linewidth=3, markersize=8, color='darkblue')
            ax_age.fill_between(np.array(steps)[valid_mask], values[valid_mask], alpha=0.3, color='darkblue')
            
            # Mark best point
            best_idx = np.argmin(values[valid_mask])
            best_step = np.array(steps)[valid_mask][best_idx]
            best_value = values[valid_mask][best_idx]
            ax_age.plot(best_step, best_value, marker='*', markersize=20, color='gold', label=f'Best: {best_value:.2f} at step {best_step}')
    
    # Add real data reference line (darkblue to match Age RMSE)
    if real_data_metrics and 'val/age_rmse' in real_data_metrics:
        real_value = real_data_metrics['val/age_rmse']
        if not np.isnan(real_value):
            ax_age.axhline(y=real_value, color='darkblue', linestyle='--', linewidth=3, alpha=0.7, label=f'Real Data ({real_value:.2f})')
    
    ax_age.set_xlabel('Training Step', fontsize=14)
    ax_age.set_ylabel('Age RMSE (years)', fontsize=14)
    ax_age.set_title('Age Prediction RMSE - Lower is Better', fontsize=16, fontweight='bold')
    ax_age.grid(True, alpha=0.3)
    ax_age.set_xlim(left=0)
    ax_age.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save Age RMSE figure
    if output_dir:
        output_path_age = Path(output_dir) / 'validation_metrics_age_rmse.png'
        plt.savefig(output_path_age, dpi=300, bbox_inches='tight')
        print(f"Saved Age RMSE plot to {output_path_age}")
    else:
        plt.savefig('validation_metrics_age_rmse.png', dpi=300, bbox_inches='tight')
        print("Saved Age RMSE plot to validation_metrics_age_rmse.png")
    
    plt.close()

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
    parser.add_argument('--real_data', type=str, default=None,
                        help='Path to real data metrics JSON file (optional)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plot (default: current directory)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plot (only save)')
    
    args = parser.parse_args()
    
    # Load metrics
    print(f"Loading metrics from {args.manifest}...")
    steps, metrics_dict = load_metrics(args.manifest)
    print(f"Loaded {len(steps)} validation steps")
    
    # Load real data metrics if provided
    real_data_metrics = None
    if args.real_data:
        print(f"Loading real data metrics from {args.real_data}...")
        real_data_metrics = load_real_data_metrics(args.real_data)
        if real_data_metrics:
            print(f"Loaded real data metrics: {list(real_data_metrics.keys())}")
        else:
            print("Warning: Could not load real data metrics")
    
    # Print summary
    print_summary(steps, metrics_dict)
    
    # Plot metrics
    print("\nGenerating plots...")
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    
    plot_metrics(steps, metrics_dict, output_dir=args.output_dir, no_show=args.no_show, real_data_metrics=real_data_metrics)

if __name__ == '__main__':
    main()

