#!/usr/bin/env python3
"""
Extract validation metrics (FID, FID RadImageNet, demographic accuracy) 
from validation manifests across all trained models in the outputs directory.

Usage:
    python extract_validation_metrics.py [--output OUTPUT_FILE] [--format csv|json]
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import csv


def find_validation_manifests(outputs_dir: str) -> List[tuple]:
    """
    Find all validation_manifest.json files in the outputs directory.
    
    Returns:
        List of tuples: (model_path, manifest_path)
    """
    manifests = []
    outputs_path = Path(outputs_dir)
    
    if not outputs_path.exists():
        print(f"Warning: Outputs directory {outputs_dir} does not exist")
        return manifests
    
    # Search for validation_manifest.json files
    for manifest_path in outputs_path.rglob("validation_manifest.json"):
        # Extract model name from path
        # e.g., outputs/output_v1.6/0_train_hcn_with_dropout_no_aux_loss/validation_manifest.json
        # -> output_v1.6/0_train_hcn_with_dropout_no_aux_loss
        relative_path = manifest_path.relative_to(outputs_path)
        model_path = str(relative_path.parent)
        
        manifests.append((model_path, manifest_path))
    
    return manifests


def extract_metrics_from_manifest(manifest_path: Path) -> List[Dict]:
    """
    Extract metrics from a validation manifest file.
    
    Returns:
        List of dictionaries, each containing metrics for one checkpoint step
    """
    results = []
    
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"Error reading {manifest_path}: {e}")
        return results
    
    for checkpoint_key, checkpoint_data in manifest.items():
        # Extract step number from checkpoint key (e.g., "checkpoint-1000" -> 1000)
        if checkpoint_key.startswith("checkpoint-"):
            try:
                step = int(checkpoint_key.split("-")[1])
            except ValueError:
                continue
        else:
            continue
        
        # Only process completed checkpoints
        if checkpoint_data.get("status") != "completed":
            continue
        
        metrics = checkpoint_data.get("metrics", {})
        
        # Extract the metrics we care about
        result = {
            "step": step,
            "fid": metrics.get("val/fid"),
            "fid_radimagenet": metrics.get("val/fid_radimagenet"),
            "sex_accuracy": metrics.get("val/sex_accuracy"),
            "race_accuracy": metrics.get("val/race_accuracy"),
            "age_rmse": metrics.get("val/age_rmse"),
        }
        
        # Only add if we have at least FID (the main metric)
        if result["fid"] is not None:
            results.append(result)
    
    # Sort by step number
    results.sort(key=lambda x: x["step"])
    
    return results


def extract_all_metrics(outputs_dir: str) -> List[Dict]:
    """
    Extract metrics from all validation manifests.
    
    Returns:
        List of dictionaries, each containing metrics for one checkpoint from one model
    """
    all_results = []
    manifests = find_validation_manifests(outputs_dir)
    
    print(f"Found {len(manifests)} validation manifest(s)")
    
    for model_path, manifest_path in manifests:
        print(f"Processing: {model_path}")
        metrics = extract_metrics_from_manifest(manifest_path)
        
        for metric in metrics:
            metric["model"] = model_path
            all_results.append(metric)
    
    return all_results


def save_to_csv(results: List[Dict], output_file: str):
    """Save results to CSV file."""
    if not results:
        print("No results to save")
        return
    
    # Get all unique keys (columns)
    columns = ["model", "step", "fid", "fid_radimagenet", "sex_accuracy", "race_accuracy", "age_rmse"]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for result in results:
            writer.writerow(result)
    
    print(f"Saved {len(results)} rows to {output_file}")


def save_to_json(results: List[Dict], output_file: str):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved {len(results)} entries to {output_file}")


def print_summary(results: List[Dict]):
    """Print a summary of the extracted metrics."""
    if not results:
        print("No results to summarize")
        return
    
    # Group by model
    models = {}
    for result in results:
        model = result["model"]
        if model not in models:
            models[model] = []
        models[model].append(result)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for model, model_results in sorted(models.items()):
        print(f"\nModel: {model}")
        print(f"  Number of checkpoints: {len(model_results)}")
        
        if model_results:
            # Find best FID (lower is better)
            fid_results = [r for r in model_results if r["fid"] is not None]
            if fid_results:
                best_fid = min(fid_results, key=lambda x: x["fid"])
                print(f"  Best FID: {best_fid['fid']:.2f} at step {best_fid['step']}")
            
            # Find best FID RadImageNet (lower is better)
            fid_rad_results = [r for r in model_results if r["fid_radimagenet"] is not None]
            if fid_rad_results:
                best_fid_rad = min(fid_rad_results, key=lambda x: x["fid_radimagenet"])
                print(f"  Best FID RadImageNet: {best_fid_rad['fid_radimagenet']:.2f} at step {best_fid_rad['step']}")
            
            # Average demographic accuracies (if available)
            sex_accs = [r["sex_accuracy"] for r in model_results if r["sex_accuracy"] is not None]
            race_accs = [r["race_accuracy"] for r in model_results if r["race_accuracy"] is not None]
            age_rmses = [r["age_rmse"] for r in model_results if r["age_rmse"] is not None]
            
            if sex_accs:
                print(f"  Average Sex Accuracy: {sum(sex_accs)/len(sex_accs):.4f}")
            if race_accs:
                print(f"  Average Race Accuracy: {sum(race_accs)/len(race_accs):.4f}")
            if age_rmses:
                print(f"  Average Age RMSE: {sum(age_rmses)/len(age_rmses):.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract validation metrics from all trained models"
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Path to outputs directory (default: outputs)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_metrics.csv",
        help="Output file path (default: validation_metrics.csv)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "json"],
        default="csv",
        help="Output format: csv or json (default: csv)"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Don't print summary statistics"
    )
    
    args = parser.parse_args()
    
    # Extract metrics
    results = extract_all_metrics(args.outputs_dir)
    
    if not results:
        print("No validation metrics found")
        return
    
    # Save results
    if args.format == "csv":
        save_to_csv(results, args.output)
    else:
        save_to_json(results, args.output)
    
    # Print summary
    if not args.no_summary:
        print_summary(results)


if __name__ == "__main__":
    main()

