import pandas as pd
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt

def get_latest_metrics_file(path, prefix):
    files = glob(f"{path}/*{prefix}*/**/*_per_class_metrics.parquet", recursive=True)
    if not files:
        return None
    # Sort by timestamp in filename if any
    return sorted(files)[-1]


def get_latest_summary_file(path, prefix, filename):
    files = glob(f"{path}/*{prefix}*/results/{filename}")
    if not files:
        return None
    return sorted(files)[-1]


def _last_scalar_from_txt(filename):
    if not filename or not os.path.exists(filename):
        return np.nan
    arr = np.loadtxt(filename)
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return float(arr[-1])
    return float(arr[-1, -1])


def _last_row_first_col_from_txt(filename):
    if not filename or not os.path.exists(filename):
        return np.nan
    arr = np.loadtxt(filename)
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return float(arr[0])
    return float(arr[-1, 0])

def main():
    results_dir = "../results"
    
    approaches = {
        'Scratch': 'scratch',
        'FT': 'jointft',
        'FT-Mem': 'jointft-mem',
        'BiC': 'bic-mem',
        'EWC': 'ewc',
        'LwF': 'lwf',
        'DER': 'der-mem',
        'EWC (SHAP W)': 'shap_weighted_ewc',
        'DER (SHAP Mem)': 'shap_exemplars_der-mem'
    }
    
    results = []
    
    for name, prefix in approaches.items():
        metrics_file = get_latest_metrics_file(results_dir, prefix)
        acc_file = get_latest_summary_file(results_dir, prefix, 'avg_accs_taw_1-*.txt')
        forg_file = get_latest_summary_file(results_dir, prefix, 'forg_taw_1-*.txt')
        if metrics_file:
            try:
                df = pd.read_parquet(metrics_file)

                def _mean_from_cell(cell):
                    arr = np.asarray(cell)
                    return float(np.nanmean(arr)) if arr.size else np.nan

                task1_f1 = np.nan
                task1_recall = np.nan
                if len(df) > 1:
                    task1_f1 = _mean_from_cell(df.iloc[1]['f1_score'])
                    task1_recall = _mean_from_cell(df.iloc[1]['recall_score'])

                task1_acc = _last_scalar_from_txt(acc_file)
                task1_forg = _last_row_first_col_from_txt(forg_file)

                results.append({
                    'Approach': name,
                    'Source Retention': _last_row_first_col_from_txt(acc_file),
                    'Task 1 Acc': task1_acc,
                    'Task 1 Forgetting': task1_forg,
                    'Task 1 Macro F1': task1_f1,
                    'Task 1 Recall': task1_recall,
                })
            except Exception as e:
                print(f"Error loading {name}: {e}")
                
    if results:
        res_df = pd.DataFrame(results)
        print("=== Comparison of Approaches ===")
        print(res_df.to_string(index=False))
        
        plt.figure(figsize=(10, 6))
        plt.bar(res_df['Approach'], res_df['Task 1 Macro F1'])
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Macro F1 Score')
        plt.title('Target Network Performance Comparison')
        plt.tight_layout()
        os.makedirs(f"{results_dir}/figures", exist_ok=True)
        plt.savefig(f"{results_dir}/figures/approach_comparison.png")
        print(f"\nSaved plot to {results_dir}/figures/approach_comparison.png")
    else:
        print("No metrics files found. Have you run the experiments?")

if __name__ == '__main__':
    main()
