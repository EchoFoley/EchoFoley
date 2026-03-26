import json
from pathlib import Path
from typing import Dict, List, Tuple
import re

def parse_timestamp(timestamp_str: str) -> float:
    """
    Parse timestamp string in "MM:SS" or "MM.SS" format to seconds.
    
    Args:
        timestamp_str: Timestamp string in "MM:SS" or "MM.SS" format
        
    Returns:
        Time in seconds
    """
    if timestamp_str is None:
        return 0.0
    
    # Handle "MM:SS" format
    if ':' in timestamp_str:
        parts = timestamp_str.split(':')
        if len(parts) == 2:
            try:
                minutes = int(parts[0])
                seconds = float(parts[1])  # Allow decimal seconds
                return minutes * 60 + seconds
            except (ValueError, IndexError):
                return 0.0
    
    # Handle decimal value; default to seconds, unless explicitly suffixed with 'm' to indicate minutes
    sanitized = timestamp_str.strip().lower()
    if sanitized.endswith('m'):
        try:
            return float(sanitized[:-1]) * 60
        except ValueError:
            return 0.0
    
    # Try to parse as float (seconds)
    try:
        return float(timestamp_str)
    except ValueError:
        return 0.0


def calculate_iou_scores(entry: Dict) -> List[float]:
    """
    Calculate temporal IoU for all correctly paired events.
    
    Args:
        entry: Dictionary containing pairs of ground truth and prediction events
        
    Returns:
        List of IoU scores
    """
    iou_scores = []
    pairs = entry['pairs']
    
    for pair in pairs:
        if pair['prediction_event'] is not None and pair['ground_truth_event'] is not None:
            gt_event = pair['ground_truth_event']
            pred_event = pair['prediction_event']
            
            # Check if timestamps exist
            if gt_event is None or pred_event is None:
                continue
            if 'start_time' not in gt_event or 'end_time' not in gt_event:
                continue
            if 'start_time' not in pred_event or 'end_time' not in pred_event:
                continue
            
            # Parse timestamps
            gt_start = parse_timestamp(gt_event.get('start_time'))
            gt_end = parse_timestamp(gt_event.get('end_time'))
            pred_start = parse_timestamp(pred_event.get('start_time'))
            pred_end = parse_timestamp(pred_event.get('end_time'))
            
            # Compute intersection
            intersection_start = max(gt_start, pred_start)
            intersection_end = min(gt_end, pred_end)
            intersection = max(0.0, intersection_end - intersection_start)
            
            # Compute union
            union_start = min(gt_start, pred_start)
            union_end = max(gt_end, pred_end)
            union = max(0.0, union_end - union_start)
            
            if union > 0:
                iou = intersection / union
            else:
                iou = 0.0
            
            iou_scores.append(iou)
    
    return iou_scores


def evaluate_file_timestamps(filepath: Path) -> Dict:
    """
    Evaluate timestamp differences for a single JSON file.
    
    Args:
        filepath: Path to the JSON results file
        
    Returns:
        Dictionary with timestamp difference statistics
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    all_ious = []
    
    for entry in data:
        iou_scores = calculate_iou_scores(entry)
        all_ious.extend(iou_scores)
    
    # Calculate statistics
    def calc_stats(values):
        if not values:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'count': 0
            }
        
        sorted_vals = sorted(values)
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values) if len(values) > 1 else 0.0
        std = variance ** 0.5
        mid = len(sorted_vals) // 2
        if len(sorted_vals) % 2 == 1:
            median = sorted_vals[mid]
        else:
            median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2 if sorted_vals else 0.0
        
        return {
            'mean': mean,
            'std': std,
            'min': min(values),
            'max': max(values),
            'median': median,
            'count': len(values)
        }
    
    return {
        'file': filepath.name,
        'num_paired_events': len(all_ious),
        'iou': calc_stats(all_ious),
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


def evaluate_model_directory_timestamps(model_dir: Path, group_by_variant: bool = False) -> Dict[str, List[Dict]]:
    """
    Evaluate timestamp differences for all JSON files in a model directory.
    
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
                
                result = evaluate_file_timestamps(json_file)
                variant_results[variant].append(result)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        # Keep only the first three runs (sorted by filename) for each variant
        for variant in variant_results:
            variant_results[variant] = sorted(variant_results[variant], key=lambda x: x['file'])[:3]
        return variant_results
    else:
        # Original behavior: return all results under a single key
        results = []
        for json_file in json_files:
            try:
                result = evaluate_file_timestamps(json_file)
                results.append(result)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        # Keep only the first three runs
        results = sorted(results, key=lambda x: x['file'])[:3]
        return {model_dir.name: results}


def calculate_statistics_across_runs(file_results: List[Dict], metric: str) -> Dict:
    """
    Calculate mean and std across runs for a specific metric.
    
    Args:
        file_results: List of results from individual files
        metric: Metric name ('start_diff', 'end_diff', or 'duration_diff')
        
    Returns:
        Dictionary with mean and std statistics
    """
    if not file_results:
        return {}
    
    # Extract metric values from all runs
    all_means = []
    all_stds = []
    all_counts = []
    
    for result in file_results:
        if metric in result:
            stats = result[metric]
            all_means.append(stats['mean'])
            all_stds.append(stats['std'])
            all_counts.append(stats['count'])
    
    if not all_means:
        return {}
    
    # Calculate overall statistics
    overall_mean = sum(all_means) / len(all_means)
    overall_std = (sum((m - overall_mean) ** 2 for m in all_means) / (len(all_means) - 1)) ** 0.5 if len(all_means) > 1 else 0.0
    total_count = sum(all_counts)
    
    return {
        'mean': overall_mean,
        'std': overall_std,
        'mean_of_means': overall_mean,
        'std_of_means': overall_std,
        'total_count': total_count,
        'num_runs': len(all_means)
    }


def aggregate_timestamp_results(file_results: List[Dict], num_runs: int = None) -> Dict:
    """
    Aggregate timestamp difference results across runs.
    
    Args:
        file_results: List of results from individual files
        num_runs: Number of runs to use (if None, uses all runs)
        
    Returns:
        Aggregated statistics
    """
    if not file_results:
        return {}
    
    # Limit to num_runs if specified
    runs_to_use = file_results[:num_runs] if num_runs else file_results
    
    if len(runs_to_use) == 0:
        return {}
    
    # Calculate statistics from per-run means and stds
    def aggregate_metric_stats(metric_name):
        all_means = []
        all_stds = []
        all_counts = []
        
        for result in runs_to_use:
            if metric_name in result:
                stats = result[metric_name]
                all_means.append(stats['mean'])
                all_stds.append(stats['std'])
                all_counts.append(stats['count'])
        
        if not all_means:
            return {}
        
        # Calculate overall mean and std
        overall_mean = sum(all_means) / len(all_means)
        overall_std = (sum((m - overall_mean) ** 2 for m in all_means) / (len(all_means) - 1)) ** 0.5 if len(all_means) > 1 else 0.0
        total_count = sum(all_counts)
        
        return {
            'mean': overall_mean,
            'std': overall_std,
            'total_count': total_count,
            'num_runs': len(all_means)
        }
    
    return {
        'num_runs': len(runs_to_use),
        'iou': aggregate_metric_stats('iou'),
    }


def plot_timestamp_results(all_results: Dict[str, List[Dict]], output_dir: Path = None):
    """
    Plot error bar charts for timestamp differences.
    
    Args:
        all_results: Dictionary mapping model names to their file results
        output_dir: Directory to save plots (if None, saves in current directory)
    """
    
    if output_dir is None:
        output_dir = Path(__file__).parent
    
    # Collect data for plotting (first three runs, include std)
    model_names = []
    iou_means = []
    iou_stds = []
    
    for model_name, file_results in sorted(all_results.items()):
        stats = aggregate_timestamp_results(file_results, num_runs=3)
        if stats and stats['num_runs'] > 0:
            iou_stats = stats.get('iou')
            if iou_stats:
                model_names.append(model_name)
                iou_means.append(iou_stats['mean'])
                iou_stds.append(iou_stats['std'])
    
    if not model_names:
        print("No data to plot!")
        return
    
    x_pos = np.arange(len(model_names))
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.bar(x_pos, iou_means, yerr=iou_stds, capsize=5, color='#2E86AB', alpha=0.8, label='IoU Mean ± Std')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temporal IoU', fontsize=12, fontweight='bold')
    ax.set_title('Temporal IoU Across Models (First 3 Runs)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    
    output_path = output_dir / 'temporal_iou.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    plt.close('all')


def print_timestamp_results(all_results: Dict[str, List[Dict]]):
    """
    Print timestamp difference results in a formatted table.
    
    Args:
        all_results: Dictionary mapping model names to their file results
    """
    print("=" * 120)
    print("TIMESTAMP DIFFERENCE EVALUATION (Ground Truth vs Prediction)")
    print("=" * 120)
    print()
    
    # Print per-run results
    print("PER-RUN RESULTS:")
    print("-" * 120)
    print(f"{'Model':<30} {'Run':<30} {'Paired':<10} {'Temporal IoU':<20}")
    print(f"{'':<30} {'':<30} {'Events':<10} {'Mean±Std':<20}")
    print("-" * 120)
    
    for model_name, file_results in sorted(all_results.items()):
        for result in sorted(file_results, key=lambda x: x['file']):
            iou_stats = result['iou']
            iou_str = f"{iou_stats['mean']:.3f}±{iou_stats['std']:.3f}" if iou_stats['count'] > 0 else "N/A"
            
            print(f"{model_name:<30} {result['file']:<30} {result['num_paired_events']:<10} "
                  f"{iou_str:<20}")
    
    print()
    print("=" * 120)
    print("AGGREGATED RESULTS ACROSS 3 RUNS (Mean ± Std):")
    print("-" * 120)
    print(f"{'Model':<30} {'Temporal IoU':<25}")
    print("-" * 120)
    
    for model_name, file_results in sorted(all_results.items()):
        # Use first 3 runs
        stats = aggregate_timestamp_results(file_results, num_runs=3)
        if stats and stats['num_runs'] > 0:
            iou_stats = stats.get('iou')
            
            if iou_stats:
                iou_str = f"{iou_stats['mean']:.3f} ± {iou_stats['std']:.3f}"
                print(f"{model_name:<30} {iou_str:<25}")
    
    print()
    print("=" * 120)
    print("DETAILED STATISTICS ACROSS 3 RUNS:")
    print("-" * 120)
    
    for model_name, file_results in sorted(all_results.items()):
        stats = aggregate_timestamp_results(file_results, num_runs=3)
        if stats and stats['num_runs'] > 0:
            iou_stats = stats.get('iou')
            
            if iou_stats:
                total_count = iou_stats.get('total_count', 0)
                print(f"\n{model_name} (across {stats['num_runs']} runs, {total_count} paired events):")
                print(f"  Temporal IoU: {iou_stats['mean']:.3f} ± {iou_stats['std']:.3f}")


def main():
    """Main function to evaluate timestamp differences for all models."""
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
            variant_results = evaluate_model_directory_timestamps(model_dir, group_by_variant=group_by_variant)
            
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
    

if __name__ == '__main__':
    main()

