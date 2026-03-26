import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def calculate_metrics(entry: Dict) -> Tuple[int, int, int, float, float]:
    """
    Calculate TP, FP, FN, Precision, and Recall for a single evaluation entry.
    
    Args:
        entry: Dictionary containing total_ground_truth_events, total_prediction_events, and pairs
        
    Returns:
        Tuple of (TP, FP, FN, Precision, Recall)
    """
    total_gt = entry['total_ground_truth_events']
    total_pred = entry['total_prediction_events']
    pairs = entry['pairs']
    
    # True Positives: pairs where prediction_event is not null
    tp = sum(1 for pair in pairs if pair['prediction_event'] is not None)
    
    # False Negatives: pairs where prediction_event is null (ground truth not matched)
    fn = sum(1 for pair in pairs if pair['prediction_event'] is None)
    
    # False Positives: predictions that don't match any ground truth
    fp = total_pred - tp
    
    # Calculate Precision and Recall
    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    
    return tp, fp, fn, precision, recall


def evaluate_file(filepath: Path) -> Dict:
    """
    Evaluate a single JSON file and return aggregated metrics.
    
    Args:
        filepath: Path to the JSON results file
        
    Returns:
        Dictionary with aggregated metrics
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_pred = 0
    num_entries = len(data)
    
    for entry in data:
        tp, fp, fn, _, _ = calculate_metrics(entry)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt += entry['total_ground_truth_events']
        total_pred += entry['total_prediction_events']
    
    # Calculate overall precision and recall
    overall_precision = total_tp / total_pred if total_pred > 0 else 0.0
    overall_recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    return {
        'file': filepath.name,
        'num_entries': num_entries,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_gt': total_gt,
        'total_pred': total_pred,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': f1_score
    }


def extract_model_variant(filename: str) -> str:
    """
    Extract model variant name from filename.
    
    Args:
        filename: JSON filename
        
    Returns:
        Model variant name or None if pattern doesn't match
    """
    # Pattern: Qwen-Qwen3-VL-{size}-A3B-{type}_{timestamp}.json
    # Pattern: Qwen-Qwen3-VL-{size}-{type}_{timestamp}.json
    # Pattern: Qwen-Qwen3-Omni-{size}-A3B-{type}_{timestamp}.json
    import re
    
    # Match Qwen3-VL variants (e.g., "30B-A3B-Instruct" or "8B-Instruct")
    match = re.search(r'Qwen-Qwen3-VL-(\d+B)(?:-A3B)?-(Instruct|Thinking)', filename)
    if match:
        size = match.group(1)
        model_type = match.group(2)
        return f"Qwen3-VL-{size}-{model_type}"
    
    # Match Qwen3-Omni variants (e.g., "30B-A3B-Instruct")
    match = re.search(r'Qwen-Qwen3-Omni-(\d+B)(?:-A3B)?-(Instruct|Thinking)', filename)
    if match:
        size = match.group(1)
        model_type = match.group(2)
        return f"Qwen3-Omni-{size}-{model_type}"
    
    return None


def evaluate_model_directory(model_dir: Path, group_by_variant: bool = False) -> Dict[str, List[Dict]]:
    """
    Evaluate all JSON files in a model directory.
    
    Args:
        model_dir: Path to the model results directory
        group_by_variant: If True, group results by model variant
        
    Returns:
        If group_by_variant is True: Dictionary mapping variant names to results
        If group_by_variant is False: Dictionary with single key containing all results
    """
    json_files = sorted(model_dir.glob('*.json'))
    
    if group_by_variant:
        # Group by variant
        variant_results = {}
        for json_file in json_files:
            try:
                variant = extract_model_variant(json_file.name)
                if variant is None:
                    # Fallback: use directory name
                    variant = model_dir.name
                
                if variant not in variant_results:
                    variant_results[variant] = []
                
                result = evaluate_file(json_file)
                variant_results[variant].append(result)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        return variant_results
    else:
        # Original behavior: return all results under a single key
        results = []
        for json_file in json_files:
            try:
                result = evaluate_file(json_file)
                results.append(result)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        return {model_dir.name: results}


def calculate_statistics_across_runs(file_results: List[Dict], num_runs: int = None) -> Dict:
    """
    Calculate mean and std across a specific number of runs.
    
    Args:
        file_results: List of results from individual files
        num_runs: Number of runs to use (if None, uses all runs)
        
    Returns:
        Dictionary with mean and std statistics
    """
    if not file_results:
        return {}
    
    # Limit to num_runs if specified
    runs_to_use = file_results[:num_runs] if num_runs else file_results
    
    if len(runs_to_use) == 0:
        return {}
    
    # Extract metrics from runs
    precisions = [r['precision'] for r in runs_to_use]
    recalls = [r['recall'] for r in runs_to_use]
    f1_scores = [r['f1_score'] for r in runs_to_use]
    
    # Calculate mean
    precision_mean = sum(precisions) / len(precisions)
    recall_mean = sum(recalls) / len(recalls)
    f1_mean = sum(f1_scores) / len(f1_scores)
    
    # Calculate standard deviation (sample std)
    if len(runs_to_use) > 1:
        precision_std = (sum((p - precision_mean) ** 2 for p in precisions) / (len(precisions) - 1)) ** 0.5
        recall_std = (sum((r - recall_mean) ** 2 for r in recalls) / (len(recalls) - 1)) ** 0.5
        f1_std = (sum((f - f1_mean) ** 2 for f in f1_scores) / (len(f1_scores) - 1)) ** 0.5
    else:
        precision_std = 0.0
        recall_std = 0.0
        f1_std = 0.0
    
    return {
        'num_runs': len(runs_to_use),
        'precision_mean': precision_mean,
        'precision_std': precision_std,
        'precision_min': min(precisions) if precisions else 0.0,
        'precision_max': max(precisions) if precisions else 0.0,
        'recall_mean': recall_mean,
        'recall_std': recall_std,
        'recall_min': min(recalls) if recalls else 0.0,
        'recall_max': max(recalls) if recalls else 0.0,
        'f1_mean': f1_mean,
        'f1_std': f1_std,
        'f1_min': min(f1_scores) if f1_scores else 0.0,
        'f1_max': max(f1_scores) if f1_scores else 0.0,
    }


def aggregate_model_results(file_results: List[Dict]) -> Dict:
    """
    Aggregate results across multiple runs of the same model.
    
    Args:
        file_results: List of results from individual files
        
    Returns:
        Aggregated statistics
    """
    if not file_results:
        return {}
    
    total_tp = sum(r['total_tp'] for r in file_results)
    total_fp = sum(r['total_fp'] for r in file_results)
    total_fn = sum(r['total_fn'] for r in file_results)
    total_gt = sum(r['total_gt'] for r in file_results)
    total_pred = sum(r['total_pred'] for r in file_results)
    
    overall_precision = total_tp / total_pred if total_pred > 0 else 0.0
    overall_recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    # Calculate per-run statistics
    precisions = [r['precision'] for r in file_results]
    recalls = [r['recall'] for r in file_results]
    f1_scores = [r['f1_score'] for r in file_results]
    
    return {
        'num_runs': len(file_results),
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_gt': total_gt,
        'total_pred': total_pred,
        'precision_mean': overall_precision,
        'precision_std': (sum((p - overall_precision) ** 2 for p in precisions) / len(precisions)) ** 0.5 if len(precisions) > 1 else 0.0,
        'precision_min': min(precisions) if precisions else 0.0,
        'precision_max': max(precisions) if precisions else 0.0,
        'recall_mean': overall_recall,
        'recall_std': (sum((r - overall_recall) ** 2 for r in recalls) / len(recalls)) ** 0.5 if len(recalls) > 1 else 0.0,
        'recall_min': min(recalls) if recalls else 0.0,
        'recall_max': max(recalls) if recalls else 0.0,
        'f1_mean': f1_score,
        'f1_std': (sum((f - f1_score) ** 2 for f in f1_scores) / len(f1_scores)) ** 0.5 if len(f1_scores) > 1 else 0.0,
        'f1_min': min(f1_scores) if f1_scores else 0.0,
        'f1_max': max(f1_scores) if f1_scores else 0.0,
    }


def print_results(all_results: Dict[str, List[Dict]]):
    """
    Print evaluation results in a formatted table.
    
    Args:
        all_results: Dictionary mapping model names to their file results
    """
    print("=" * 100)
    print("SOUNDING EVENT DETECTION EVALUATION RESULTS")
    print("=" * 100)
    print()
    
    # Print per-run results
    print("PER-RUN RESULTS:")
    print("-" * 100)
    print(f"{'Model':<30} {'Run':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
    print("-" * 100)
    
    for model_name, file_results in sorted(all_results.items()):
        for result in sorted(file_results, key=lambda x: x['file']):
            print(f"{model_name:<30} {result['file']:<20} "
                  f"{result['precision']:<12.4f} {result['recall']:<12.4f} "
                  f"{result['f1_score']:<12.4f} {result['total_tp']:<8} "
                  f"{result['total_fp']:<8} {result['total_fn']:<8}")
    
    print()
    print("=" * 100)
    print("AGGREGATED RESULTS BY MODEL:")
    print("-" * 100)
    print(f"{'Model':<30} {'Runs':<8} {'Precision':<20} {'Recall':<20} {'F1-Score':<20}")
    print(f"{'':<30} {'':<8} {'Mean±Std (Min-Max)':<20} {'Mean±Std (Min-Max)':<20} {'Mean±Std (Min-Max)':<20}")
    print("-" * 100)
    
    for model_name, file_results in sorted(all_results.items()):
        agg = aggregate_model_results(file_results)
        if agg:
            prec_str = f"{agg['precision_mean']:.4f}±{agg['precision_std']:.4f} ({agg['precision_min']:.4f}-{agg['precision_max']:.4f})"
            rec_str = f"{agg['recall_mean']:.4f}±{agg['recall_std']:.4f} ({agg['recall_min']:.4f}-{agg['recall_max']:.4f})"
            f1_str = f"{agg['f1_mean']:.4f}±{agg['f1_std']:.4f} ({agg['f1_min']:.4f}-{agg['f1_max']:.4f})"
            print(f"{model_name:<30} {agg['num_runs']:<8} {prec_str:<20} {rec_str:<20} {f1_str:<20}")
    
    print()
    print("=" * 100)
    print("STATISTICS ACROSS 3 RUNS (Mean ± Std):")
    print("-" * 100)
    print(f"{'Model':<30} {'Precision':<25} {'Recall':<25} {'F1-Score':<25}")
    print("-" * 100)
    
    for model_name, file_results in sorted(all_results.items()):
        # Use first 3 runs for each model
        stats = calculate_statistics_across_runs(file_results, num_runs=3)
        if stats and stats['num_runs'] > 0:
            prec_str = f"{stats['precision_mean']:.4f} ± {stats['precision_std']:.4f}"
            rec_str = f"{stats['recall_mean']:.4f} ± {stats['recall_std']:.4f}"
            f1_str = f"{stats['f1_mean']:.4f} ± {stats['f1_std']:.4f}"
            print(f"{model_name:<30} {prec_str:<25} {rec_str:<25} {f1_str:<25}")
    
    print()
    print("=" * 100)
    print("DETAILED STATISTICS ACROSS 3 RUNS:")
    print("-" * 100)
    
    for model_name, file_results in sorted(all_results.items()):
        stats = calculate_statistics_across_runs(file_results, num_runs=3)
        if stats and stats['num_runs'] > 0:
            print(f"\n{model_name} (across {stats['num_runs']} runs):")
            print(f"  Precision: {stats['precision_mean']:.4f} ± {stats['precision_std']:.4f} (range: {stats['precision_min']:.4f} - {stats['precision_max']:.4f})")
            print(f"  Recall: {stats['recall_mean']:.4f} ± {stats['recall_std']:.4f} (range: {stats['recall_min']:.4f} - {stats['recall_max']:.4f})")
            print(f"  F1-Score: {stats['f1_mean']:.4f} ± {stats['f1_std']:.4f} (range: {stats['f1_min']:.4f} - {stats['f1_max']:.4f})")
    
    print()
    print("=" * 100)
    print("SUMMARY STATISTICS (All Runs):")
    print("-" * 100)
    
    for model_name, file_results in sorted(all_results.items()):
        agg = aggregate_model_results(file_results)
        if agg:
            print(f"\n{model_name}:")
            print(f"  Total Ground Truth Events: {agg['total_gt']}")
            print(f"  Total Predicted Events: {agg['total_pred']}")
            print(f"  True Positives: {agg['total_tp']}")
            print(f"  False Positives: {agg['total_fp']}")
            print(f"  False Negatives: {agg['total_fn']}")
            print(f"  Precision: {agg['precision_mean']:.4f} ± {agg['precision_std']:.4f}")
            print(f"  Recall: {agg['recall_mean']:.4f} ± {agg['recall_std']:.4f}")
            print(f"  F1-Score: {agg['f1_mean']:.4f} ± {agg['f1_std']:.4f}")


def main():
    """Main function to evaluate all models."""
    base_dir = Path(__file__).parent
    
    # Define model directories
    model_dirs = {
        'Gemini Flash': base_dir / 'results_gemini_flash',
        'Gemini Pro': base_dir / 'results_gemini_pro',
        'Qwen3-VL Zero-Shot': base_dir / 'results_qwen3vl_zero_shot',
        'Qwen3-Omni Zero-Shot': base_dir / 'results_omini_zero_shot',
        'OmniVinci Zero-Shot': base_dir / 'results_omnivinci_zero_shot',
    }
    
    all_results = {}
    
    for model_name, model_dir in model_dirs.items():
        if model_dir.exists():
            # Group by variant for Qwen models
            group_by_variant = 'Qwen' in model_name
            variant_results = evaluate_model_directory(model_dir, group_by_variant=group_by_variant)
            
            if variant_results:
                if group_by_variant:
                    # Add each variant as a separate model
                    for variant_name, file_results in variant_results.items():
                        all_results[variant_name] = file_results
                else:
                    # Use the original model name
                    all_results[model_name] = list(variant_results.values())[0]
        else:
            print(f"Warning: Directory {model_dir} does not exist")
    
    if all_results:
        print_results(all_results)
    else:
        print("No results found!")


if __name__ == '__main__':
    main()

