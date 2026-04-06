"""
Statistical Validation for Multi-Seed Results
Performs McNemar's test, bootstrap CI, and significance analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class StatisticalValidator:
    def __init__(self, results_dir='results/multiseed'):
        self.results_dir = Path(results_dir)
        self.results = {}
        
    def load_experiment_results(self, experiment_name: str) -> List[Dict]:
        """Load all seed results for an experiment"""
        exp_dir = self.results_dir / experiment_name
        seed_results = []
        
        for seed_dir in sorted(exp_dir.glob('seed_*')):
            result_file = seed_dir / 'test_metrics.json'
            if result_file.exists():
                with open(result_file, 'r') as f:
                    result = json.load(f)
                    result['seed'] = int(seed_dir.name.split('_')[1])
                    seed_results.append(result)
        
        self.results[experiment_name] = seed_results
        return seed_results
    
    def compute_statistics(self, experiment_name: str, metric: str) -> Dict:
        """Compute mean, std, CI for a metric"""
        if experiment_name not in self.results:
            self.load_experiment_results(experiment_name)
        
        values = [r[metric] for r in self.results[experiment_name]]
        
        if not values:
            return {
                'mean': 0.0, 'std': 0.0, 'median': 0.0,
                'min': 0.0, 'max': 0.0,
                'ci_95_lower': 0.0, 'ci_95_upper': 0.0,
                'n_seeds': 0
            }
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        # Bootstrap confidence interval
        ci_lower, ci_upper = self.bootstrap_ci(values)
        
        return {
            'mean': mean,
            'std': std,
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'n_seeds': len(values)
        }
    
    def bootstrap_ci(self, values: List[float], n_iterations: int = 1000, 
                     ci: float = 0.95) -> Tuple[float, float]:
        """Compute bootstrap confidence intervals"""
        bootstrap_means = []
        n = len(values)
        
        for _ in range(n_iterations):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = (1 - ci) / 2
        lower = np.percentile(bootstrap_means, alpha * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
        
        return lower, upper
    
    def mcnemar_test(self, exp1_name: str, exp2_name: str) -> Dict:
        """
        McNemar's test for paired predictions
        Requires: predictions saved for each seed
        """
        # Load predictions for both experiments
        exp1_preds = self._load_predictions(exp1_name)
        exp2_preds = self._load_predictions(exp2_name)
        
        if exp1_preds is None or exp2_preds is None:
            return {'error': 'Predictions not found'}
        
        # Build contingency table
        n_01 = np.sum((exp1_preds == self.true_labels) & 
                      (exp2_preds != self.true_labels))
        n_10 = np.sum((exp1_preds != self.true_labels) & 
                      (exp2_preds == self.true_labels))
        
        # McNemar's test statistic (with continuity correction)
        statistic = (abs(n_01 - n_10) - 1)**2 / (n_01 + n_10) if (n_01 + n_10) > 0 else 0
        p_value = 1 - stats.chi2.cdf(statistic, 1)
        
        return {
            'n_01': int(n_01),
            'n_10': int(n_10),
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01
        }
    
    def wilcoxon_test(self, exp1_name: str, exp2_name: str, metric: str) -> Dict:
        """Wilcoxon signed-rank test for paired samples"""
        if exp1_name not in self.results:
            self.load_experiment_results(exp1_name)
        if exp2_name not in self.results:
            self.load_experiment_results(exp2_name)
        
        # Get metric values
        values1 = [r[metric] for r in self.results[exp1_name]]
        values2 = [r[metric] for r in self.results[exp2_name]]
        
        # Ensure same number of seeds
        min_len = min(len(values1), len(values2))
        values1 = values1[:min_len]
        values2 = values2[:min_len]
        
        statistic, p_value = wilcoxon(values1, values2)
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01,
            'mean_diff': np.mean(values1) - np.mean(values2)
        }
    
    def bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """Bonferroni correction for multiple comparisons"""
        n = len(p_values)
        corrected = [min(p * n, 1.0) for p in p_values]
        return corrected
    
    def generate_comparison_table(self, experiments: List[str], 
                                   metric: str = 'f1') -> pd.DataFrame:
        """Generate LaTeX-ready comparison table"""
        rows = []
        
        for exp_name in experiments:
            stats = self.compute_statistics(exp_name, metric)
            rows.append({
                'Method': exp_name.replace('_', ' ').title(),
                'Mean': f"{stats['mean']:.3f}",
                'Std': f"{stats['std']:.3f}",
                '95% CI': f"[{stats['ci_95_lower']:.3f}, {stats['ci_95_upper']:.3f}]",
                'Median': f"{stats['median']:.3f}",
                'Seeds': stats['n_seeds']
            })
        
        df = pd.DataFrame(rows)
        return df
    
    def generate_publication_tables(self, output_dir: str = 'results/publication'):
        """Generate all publication-ready tables"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        experiments = ['curriculum', 'baseline', 'disagreement']
        metrics = ['f1', 's_recall', 'v_recall', 'precision']
        
        for metric in metrics:
            df = self.generate_comparison_table(experiments, metric)
            
            # Save as CSV
            csv_file = output_dir / f'table_{metric}_comparison.csv'
            df.to_csv(csv_file, index=False)
            
            # Save as LaTeX
            latex_file = output_dir / f'table_{metric}_comparison.tex'
            latex = df.to_latex(index=False, escape=False)
            with open(latex_file, 'w') as f:
                f.write(latex)
            
            print(f"✓ Generated {metric} comparison table")
    
    def _load_predictions(self, experiment_name: str) -> np.ndarray:
        """Load predictions from experiment (seed 42 by default)"""
        pred_file = self.results_dir / experiment_name / 'seed_42' / 'predictions.npy'
        if pred_file.exists():
            return np.load(pred_file)
        return None
    
    def generate_significance_matrix(self, experiments: List[str], 
                                      metric: str = 'f1') -> pd.DataFrame:
        """Generate pairwise significance test matrix"""
        n = len(experiments)
        p_matrix = np.zeros((n, n))
        
        for i, exp1 in enumerate(experiments):
            for j, exp2 in enumerate(experiments):
                if i != j:
                    result = self.wilcoxon_test(exp1, exp2, metric)
                    p_matrix[i, j] = result['p_value']
        
        df = pd.DataFrame(
            p_matrix, 
            index=experiments, 
            columns=experiments
        )
        
        return df
    
    def run_full_validation(self, experiments: List[str]):
        """Run complete statistical validation pipeline"""
        print("\n" + "="*60)
        print("STATISTICAL VALIDATION REPORT")
        print("="*60 + "\n")
        
        # Load all results
        for exp in experiments:
            self.load_experiment_results(exp)
        
        # 1. Descriptive statistics
        print("1. DESCRIPTIVE STATISTICS")
        print("-" * 60)
        for metric in ['f1', 's_recall', 'precision']:
            print(f"\n{metric.upper()}:")
            for exp in experiments:
                stats = self.compute_statistics(exp, metric)
                print(f"  {exp:20s}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                      f"[{stats['ci_95_lower']:.3f}, {stats['ci_95_upper']:.3f}]")
        
        # 2. Pairwise comparisons
        print("\n\n2. PAIRWISE SIGNIFICANCE TESTS (Wilcoxon)")
        print("-" * 60)
        for i, exp1 in enumerate(experiments):
            for exp2 in experiments[i+1:]:
                result = self.wilcoxon_test(exp1, exp2, 'f1')
                sig = "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "ns"
                print(f"{exp1} vs {exp2}: p={result['p_value']:.4f} {sig}")
        
        # 3. Generate tables
        print("\n\n3. GENERATING PUBLICATION TABLES")
        print("-" * 60)
        self.generate_publication_tables()
        
        print("\n✓ Statistical validation complete!")

def main():
    validator = StatisticalValidator()
    
    experiments = ['curriculum', 'baseline', 'disagreement']
    validator.run_full_validation(experiments)

if __name__ == '__main__':
    main()
