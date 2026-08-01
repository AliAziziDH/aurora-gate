"""Visualize error analysis results with confusion matrix and performance charts."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_reports():
    """Load error analysis reports."""
    error_report_path = Path('experiments/error_analysis.json')
    detailed_report_path = Path('experiments/error_analysis_detailed.json')
    
    with error_report_path.open('r') as f:
        error_report = json.load(f)
    
    with detailed_report_path.open('r') as f:
        detailed_report = json.load(f)
    
    return error_report, detailed_report

def plot_confusion_matrix(error_report):
    """Plot normalized confusion matrix."""
    labels = error_report['confusion_matrix']['labels']
    raw_matrix = np.array(error_report['confusion_matrix']['raw'])
    
    # Normalize by row (true label)
    row_sums = raw_matrix.sum(axis=1, keepdims=True)
    norm_matrix = raw_matrix / row_sums
    
    plt.figure(figsize=(12, 10))
    
    # Create mask for diagonal (correct classifications)
    mask = np.zeros_like(norm_matrix)
    mask[np.diag_indices_from(mask)] = 1
    
    # Plot heatmap
    ax = sns.heatmap(
        norm_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        mask=mask,
        vmin=0,
        vmax=0.5,
        linewidths=.5
    )
    
    # Add diagonal annotations (correct classification rates)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                ax.text(j + 0.5, i + 0.5, f"{norm_matrix[i, j]:.2f}",
                       ha='center', va='center', color='green', fontweight='bold')
    
    plt.title('Normalized Confusion Matrix (Row = True Label, Column = Predicted Label)', pad=20)
    plt.xlabel('Predicted Category', labelpad=15)
    plt.ylabel('True Category', labelpad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save and show
    plt.savefig('experiments/confusion_matrix_visual.png', dpi=150, bbox_inches='tight')
    print("✅ Confusion matrix saved to: experiments/confusion_matrix_visual.png")
    plt.close()

def plot_per_class_performance(detailed_report):
    """Plot per-class performance metrics."""
    metrics = pd.DataFrame(detailed_report['per_class_metrics']).T
    metrics['category'] = metrics.index
    metrics = metrics.sort_values('f1', ascending=False)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # Plot F1 scores
    sns.barplot(x='f1', y='category', data=metrics, ax=axes[0], palette='viridis')
    axes[0].set_title('F1 Score by Category')
    axes[0].set_xlabel('F1 Score')
    axes[0].set_ylabel('')
    axes[0].set_xlim(0.8, 1.0)
    
    # Plot Precision vs Recall
    sns.scatterplot(x='precision', y='recall', hue='f1', size='support', 
                   data=metrics, ax=axes[1], palette='viridis', sizes=(100, 500))
    axes[1].set_title('Precision vs Recall')
    axes[1].set_xlabel('Precision')
    axes[1].set_ylabel('Recall')
    axes[1].set_xlim(0.8, 1.0)
    axes[1].set_ylim(0.8, 1.0)
    
    # Add ideal line
    axes[1].plot([0.8, 1.0], [0.8, 1.0], 'r--', alpha=0.3)
    
    # Plot Error Rates
    error_rates = []
    for category, metrics_data in detailed_report['per_class_metrics'].items():
        # Calculate error rate from false positives and false negatives
        total_errors = metrics_data['false_positives'] + metrics_data['false_negatives']
        support = metrics_data['support']
        error_rate = total_errors / (support + total_errors) if (support + total_errors) > 0 else 0
        error_rates.append({
            'category': category,
            'error_rate': error_rate,
            'f1': metrics_data['f1']
        })
    
    error_df = pd.DataFrame(error_rates).sort_values('error_rate', ascending=False)
    sns.barplot(x='error_rate', y='category', data=error_df, ax=axes[2], palette='Reds_r')
    axes[2].set_title('Error Rate by Category')
    axes[2].set_xlabel('Error Rate')
    axes[2].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig('experiments/per_class_performance.png', dpi=150, bbox_inches='tight')
    print("✅ Per-class performance charts saved to: experiments/per_class_performance.png")
    plt.close()

def plot_confusion_pairs(detailed_report):
    """Plot most confused category pairs."""
    confused_pairs = detailed_report['confusion_patterns']['most_confused_pairs'][:10]
    
    pairs = [item['pair'] for item in confused_pairs]
    scores = [item['confusion_score'] for item in confused_pairs]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=scores, y=pairs, palette='coolwarm')
    
    plt.title('Top 10 Most Confused Category Pairs', pad=20)
    plt.xlabel('Total Confusion Score', labelpad=15)
    plt.ylabel('Category Pair', labelpad=15)
    
    # Add value labels
    for i, score in enumerate(scores):
        plt.text(score + 1, i, str(score), va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('experiments/confusion_pairs.png', dpi=150, bbox_inches='tight')
    print("✅ Confusion pairs chart saved to: experiments/confusion_pairs.png")
    plt.close()

def plot_fold_performance(error_report):
    """Plot performance across folds."""
    folds = [fold['fold'] for fold in error_report['folds']]
    f1_scores = [fold['macro_f1'] for fold in error_report['folds']]
    rows = [fold['rows'] for fold in error_report['folds']]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot F1 scores
    color = 'tab:blue'
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Macro F1 Score', color=color)
    bars = ax1.bar(folds, f1_scores, color=color, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add value labels
    for i, score in enumerate(f1_scores):
        ax1.text(i + 1, score + 0.002, f'{score:.4f}', ha='center', va='bottom')
    
    # Add horizontal line for average
    avg_f1 = np.mean(f1_scores)
    ax1.axhline(avg_f1, color='red', linestyle='--', alpha=0.5, label=f'Average: {avg_f1:.4f}')
    
    plt.title('Performance Across Validation Folds', pad=20)
    plt.xticks(folds)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/fold_performance.png', dpi=150, bbox_inches='tight')
    print("✅ Fold performance chart saved to: experiments/fold_performance.png")
    plt.close()

def generate_html_report(detailed_report):
    """Generate HTML report with all findings."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AuroraGate Error Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; color: #333; }}
            h1 {{ color: #2c3e50; text-align: center; }}
            h2 {{ color: #3498db; }}
            .metrics {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .highlight {{ background: #fff3cd; padding: 10px; border-radius: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #3498db; color: white; }}
            tr:hover {{ background: #f5f5f5; }}
            .good {{ color: #27ae60; font-weight: bold; }}
            .bad {{ color: #e74c3c; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>AuroraGate Error Analysis Report</h1>
        
        <h2>📊 Overall Performance</h2>
        <div class="metrics">
            <p><strong>Macro F1 Score:</strong> <span class="good">{detailed_report['overall_metrics']['macro_f1']:.4f}</span> ± {detailed_report['overall_metrics']['f1_std']:.4f}</p>
            <p><strong>Total Validation Rows:</strong> {detailed_report['overall_metrics']['total_validation_rows']}</p>
            <p><strong>Average Fold F1:</strong> {detailed_report['overall_metrics']['average_fold_f1']:.4f}</p>
        </div>
        
        <h2>📉 Lowest-F1 Classes</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Category</th>
                <th>F1 Score</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Support</th>
                <th>False Negatives</th>
                <th>False Positives</th>
            </tr>
    """
    
    # Add lowest F1 classes
    for i, cls in enumerate(detailed_report['lowest_f1_classes'], 1):
        html_content += f"""
            <tr>
                <td>{i}</td>
                <td><strong>{cls['category']}</strong></td>
                <td>{cls['f1']:.4f}</td>
                <td>{cls['precision']:.4f}</td>
                <td>{cls['recall']:.4f}</td>
                <td>{cls['support']}</td>
                <td class="bad">{cls['false_negatives']}</td>
                <td class="bad">{cls['false_positives']}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>🔀 Top Misclassifications</h2>
        <div class="highlight">
            <p>These are the most common misclassification patterns:</p>
            <ol>
    """
    
    # Add top misclassifications
    for i, item in enumerate(detailed_report['confusion_patterns']['top_misclassifications'][:10], 1):
        html_content += f"<li>{item['pattern']} ({item['count']} times)</li>\n"
    
    html_content += """
            </ol>
        </div>
        
        <h2>💡 Suggested Improvements</h2>
        <div class="highlight">
            <p>Based on error patterns, these improvements are suggested:</p>
            <ol>
    """
    
    # Add suggested improvements
    for i, suggestion in enumerate(detailed_report['suggested_improvements'][:5], 1):
        rule = suggestion.get('rule', 'No rule')
        category = suggestion.get('suggested_category', 'N/A')
        reason = suggestion.get('reason', 'No reason provided')
        html_content += f"<li><strong>{rule}</strong> → {category}<br><em>{reason}</em></li>\n"
    
    html_content += """
            </ol>
        </div>
        
        <h2>📈 Full Performance Metrics</h2>
        <table>
            <tr>
                <th>Category</th>
                <th>F1</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Support</th>
                <th>Error Rate</th>
            </tr>
    """
    
    # Add full metrics
    for category, metrics_data in sorted(detailed_report['per_class_metrics'].items(), key=lambda x: x[1]['f1'], reverse=True):
        # Calculate error rate
        total_errors = metrics_data['false_positives'] + metrics_data['false_negatives']
        support = metrics_data['support']
        error_rate = total_errors / (support + total_errors) if (support + total_errors) > 0 else 0
        error_class = 'good' if error_rate < 0.1 else 'bad'
        error_class = 'good' if error_rate < 0.1 else 'bad'
        # Calculate error rate for HTML report
        total_errors = metrics_data['false_positives'] + metrics_data['false_negatives']
        support = metrics_data['support']
        error_rate = total_errors / (support + total_errors) if (support + total_errors) > 0 else 0
        error_class = 'good' if error_rate < 0.1 else 'bad'
        html_content += f"""
            <tr>
                <td><strong>{category}</strong></td>
                <td>{metrics_data['f1']:.4f}</td>
                <td>{metrics_data['precision']:.4f}</td>
                <td>{metrics_data['recall']:.4f}</td>
                <td>{metrics_data['support']}</td>
                <td class="{error_class}">{error_rate:.3f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>📊 Visualizations</h2>
        <div class="highlight">
            <p>Generated visualizations:</p>
            <ul>
                <li><a href="confusion_matrix_visual.png">Confusion Matrix</a></li>
                <li><a href="per_class_performance.png">Per-Class Performance</a></li>
                <li><a href="confusion_pairs.png">Confusion Pairs</a></li>
                <li><a href="fold_performance.png">Fold Performance</a></li>
            </ul>
        </div>
        
        <div style="text-align: center; margin-top: 30px; color: #7f8c8d;">
            <p>Generated by AuroraGate Error Analysis System</p>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    html_path = Path('experiments/error_analysis_report.html')
    with html_path.open('w') as f:
        f.write(html_content)
    
    print("✅ HTML report saved to: experiments/error_analysis_report.html")

def main():
    """Generate all visualizations and reports."""
    print("📊 Generating AuroraGate Error Analysis Visualizations")
    print("=" * 60)
    
    # Load reports
    error_report, detailed_report = load_reports()
    
    # Generate visualizations
    plot_confusion_matrix(error_report)
    plot_per_class_performance(detailed_report)
    plot_confusion_pairs(detailed_report)
    plot_fold_performance(error_report)
    
    # Generate HTML report
    generate_html_report(detailed_report)
    
    print("\n" + "=" * 60)
    print("✅ All visualizations and reports generated successfully!")
    print("=" * 60)
    print("\n📁 Generated files:")
    print("   • experiments/confusion_matrix_visual.png")
    print("   • experiments/per_class_performance.png")
    print("   • experiments/confusion_pairs.png")
    print("   • experiments/fold_performance.png")
    print("   • experiments/error_analysis_report.html")
    print("\n💡 Key Findings:")
    print(f"   • Overall F1: {detailed_report['overall_metrics']['macro_f1']:.4f}")
    print(f"   • Lowest F1: {detailed_report['lowest_f1_classes'][0]['category']} ({detailed_report['lowest_f1_classes'][0]['f1']:.4f})")
    print(f"   • Top confusion: {detailed_report['confusion_patterns']['top_misclassifications'][0]['pattern']}")

if __name__ == '__main__':
    main()