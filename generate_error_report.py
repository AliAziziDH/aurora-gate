"""Generate a comprehensive error analysis report from the error analysis results."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

def load_error_analysis():
    """Load the error analysis report."""
    report_path = Path('experiments/error_analysis.json')
    with report_path.open('r') as f:
        return json.load(f)

def calculate_overall_f1(report):
    """Calculate overall macro F1 from fold scores."""
    fold_scores = [fold['macro_f1'] for fold in report['folds']]
    return np.mean(fold_scores)

def find_confusion_patterns(confusion_matrix, labels):
    """Find the most common misclassification patterns."""
    raw_matrix = np.array(confusion_matrix['raw'])
    patterns = []
    
    # Find off-diagonal elements (misclassifications)
    for true_idx in range(len(labels)):
        for pred_idx in range(len(labels)):
            if true_idx != pred_idx:
                count = raw_matrix[true_idx, pred_idx]
                if count > 0:
                    patterns.append((
                        f"{labels[true_idx]} → {labels[pred_idx]}",
                        count
                    ))
    
    # Sort by frequency
    patterns.sort(key=lambda x: x[1], reverse=True)
    return patterns

def find_most_confused_pairs(confusion_matrix, labels):
    """Find pairs of categories that are most commonly confused."""
    raw_matrix = np.array(confusion_matrix['raw'])
    confused_pairs = []
    
    # Calculate confusion scores for each pair
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            # Total misclassifications between i and j
            confusion_score = raw_matrix[i, j] + raw_matrix[j, i]
            if confusion_score > 0:
                confused_pairs.append((
                    f"{labels[i]} ↔ {labels[j]}",
                    confusion_score
                ))
    
    # Sort by confusion frequency
    confused_pairs.sort(key=lambda x: x[1], reverse=True)
    return confused_pairs

def analyze_error_patterns(report, validation_data_path='experiments/validation_data.parquet'):
    """Analyze patterns in misclassified transactions."""
    patterns = {}
    
    # Load validation data if available
    if Path(validation_data_path).exists():
        val_data = pd.read_parquet(validation_data_path)
        misclassified = val_data[val_data['actual'] != val_data['predicted']]
        
        # Analyze by amount
        if 'amount' in misclassified.columns:
            patterns['amount'] = {
                'low': len(misclassified[misclassified['amount'] < 25]),
                'medium': len(misclassified[(misclassified['amount'] >= 25) & (misclassified['amount'] < 100)]),
                'high': len(misclassified[misclassified['amount'] >= 100])
            }
        
        # Analyze by day of week
        if 'day_of_week' in misclassified.columns:
            patterns['day_of_week'] = misclassified['day_of_week'].value_counts().to_dict()
        
        # Analyze by store frequency
        if 'store_frequency' in misclassified.columns:
            patterns['store_frequency'] = {
                'single': len(misclassified[misclassified['store_frequency'] == 1]),
                'repeat': len(misclassified[misclassified['store_frequency'] > 1])
            }
    
    return patterns

def generate_detailed_report():
    """Generate a comprehensive error analysis report."""
    report = load_error_analysis()
    
    # Calculate overall metrics
    overall_f1 = calculate_overall_f1(report)
    total_rows = sum(fold['rows'] for fold in report['folds'])
    
    # Find confusion patterns
    labels = report['confusion_matrix']['labels']
    raw_matrix = np.array(report['confusion_matrix']['raw'])
    
    # Calculate per-class metrics
    per_class = report['per_class_metrics']
    
    # Find lowest F1 classes
    lowest_f1_classes = sorted(per_class.items(), key=lambda x: x[1]['f1'])[:3]
    
    # Find most common misclassifications
    confusion_patterns = find_confusion_patterns(report['confusion_matrix'], labels)[:10]
    
    # Find most confused pairs
    confused_pairs = find_most_confused_pairs(report['confusion_matrix'], labels)[:5]
    
    # Analyze error patterns
    error_patterns = analyze_error_patterns(report)
    
    # Generate detailed report
    detailed_report = {
        'overall_metrics': {
            'macro_f1': float(overall_f1),
            'total_validation_rows': total_rows,
            'average_fold_f1': float(np.mean([fold['macro_f1'] for fold in report['folds']])),
            'f1_std': float(np.std([fold['macro_f1'] for fold in report['folds']]))
        },
        'lowest_f1_classes': [
            {
                'category': category,
                'f1': float(metrics['f1']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'false_negatives': int(metrics['false_negatives']),
                'false_positives': int(metrics['false_positives']),
                'support': int(metrics['support'])
            }
            for category, metrics in lowest_f1_classes
        ],
        'confusion_patterns': {
            'top_misclassifications': [
                {'pattern': pattern, 'count': int(count)}
                for pattern, count in confusion_patterns
            ],
            'most_confused_pairs': [
                {'pair': pair, 'confusion_score': int(score)}
                for pair, score in confused_pairs
            ]
        },
        'per_class_metrics': {
            category: {
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1']),
                'support': int(metrics['support']),
                'false_negatives': int(metrics['false_negatives']),
                'false_positives': int(metrics['false_positives'])
            }
            for category, metrics in per_class.items()
        },
        'error_patterns': error_patterns,
        'suggested_improvements': report.get('suggested_rules', []),
        'model_info': report['model']
    }
    
    # Save detailed report
    detailed_report_path = Path('experiments/error_analysis_detailed.json')
    with detailed_report_path.open('w') as f:
        json.dump(detailed_report, f, indent=2)
    
    return detailed_report

def print_summary_report(detailed_report):
    """Print a summary of the error analysis."""
    print('=' * 80)
    print('AURORAGATE DETAILED ERROR ANALYSIS REPORT')
    print('=' * 80)
    
    # Overall metrics
    print(f'\n📊 OVERALL PERFORMANCE')
    print(f'   Macro F1 Score: {detailed_report["overall_metrics"]["macro_f1"]:.4f} ± {detailed_report["overall_metrics"]["f1_std"]:.4f}')
    print(f'   Total Validation Rows: {detailed_report["overall_metrics"]["total_validation_rows"]}')
    print(f'   Average Fold F1: {detailed_report["overall_metrics"]["average_fold_f1"]:.4f}')
    
    # Lowest F1 classes
    print(f'\n📉 LOWEST-F1 CLASSES (Top 3)')
    for i, cls in enumerate(detailed_report['lowest_f1_classes'], 1):
        print(f'   {i}. {cls["category"]}:')
        print(f'      F1: {cls["f1"]:.4f} | Precision: {cls["precision"]:.4f} | Recall: {cls["recall"]:.4f}')
        print(f'      Support: {cls["support"]} | FN: {cls["false_negatives"]} | FP: {cls["false_positives"]}')
    
    # Confusion patterns
    print(f'\n🔀 TOP MISCLASSIFICATIONS')
    for i, item in enumerate(detailed_report['confusion_patterns']['top_misclassifications'][:5], 1):
        print(f'   {i}. {item["pattern"]}: {item["count"]} times')
    
    print(f'\n🔀 MOST CONFUSED CATEGORY PAIRS')
    for i, item in enumerate(detailed_report['confusion_patterns']['most_confused_pairs'][:3], 1):
        print(f'   {i}. {item["pair"]}: {item["confusion_score"]} total confusions')
    
    # Error patterns
    if detailed_report['error_patterns']:
        print(f'\n📊 ERROR PATTERNS BY FEATURE')
        if 'amount' in detailed_report['error_patterns']:
            print(f'   By Amount:')
            for amount_range, count in detailed_report['error_patterns']['amount'].items():
                print(f'      {amount_range}: {count} misclassifications')
        if 'day_of_week' in detailed_report['error_patterns']:
            print(f'   By Day of Week:')
            for day, count in sorted(detailed_report['error_patterns']['day_of_week'].items(), key=lambda x: x[1], reverse=True):
                print(f'      {day}: {count} misclassifications')
    
    # Suggested improvements
    if detailed_report['suggested_improvements']:
        print(f'\n💡 SUGGESTED IMPROVEMENTS')
        for i, suggestion in enumerate(detailed_report['suggested_improvements'][:5], 1):
            print(f'   {i}. {suggestion.get("rule", "No rule")} → {suggestion.get("suggested_category", "N/A")}')
            if 'reason' in suggestion:
                print(f'      Reason: {suggestion["reason"]}')
    
    print(f'\n📈 FULL PER-CLASS METRICS')
    print(f'   Category'.ljust(25) + 'F1'.rjust(8) + 'Prec'.rjust(8) + 'Rec'.rjust(8) + 'Supp'.rjust(6))
    for category, metrics in sorted(detailed_report['per_class_metrics'].items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f'   {category.ljust(25)} {metrics["f1"]:.4f}  {metrics["precision"]:.4f}  {metrics["recall"]:.4f}  {metrics["support"]:4d}')
    
    print('=' * 80)
    print(f'📄 Full report saved to: experiments/error_analysis_detailed.json')
    print('=' * 80)

if __name__ == '__main__':
    # Generate detailed report
    detailed_report = generate_detailed_report()
    
    # Print summary
    print_summary_report(detailed_report)
    
    print(f'\n✅ Error analysis report generated successfully!')