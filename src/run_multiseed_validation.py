"""
60-Hour Multi-Seed Validation Runner
Executes multi-seed experiments for statistical validation
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import argparse

# Experiment Configuration
EXPERIMENTS = {
    'curriculum': {
        'script': 'train_curriculum.py',
        'seeds': [42, 123, 456, 789, 1024, 2048, 3141, 5678, 8192, 9999],
        'epochs': 100,
        'priority': 1,  # CRITICAL
        'time_estimate': 2.0  # hours per seed
    },
    'baseline': {
        'script': 'train_fusion_single.py',
        'seeds': [42, 123, 456, 789, 1024],
        'epochs': 100,
        'priority': 2,  # HIGH
        'time_estimate': 2.0
    },
    'disagreement': {
        'script': 'train_disagreement.py',
        'seeds': [42, 123, 456, 789, 1024],
        'epochs': 100,
        'priority': 3,  # MEDIUM
        'time_estimate': 2.0
    },
    'curriculum_attention': {
        'script': 'train_curriculum_attention.py',
        'seeds': [42, 123, 456],
        'epochs': 100,
        'priority': 4,  # EXPERIMENTAL
        'time_estimate': 2.5
    },
    'curriculum_contrastive': {
        'script': 'train_curriculum_contrastive.py',
        'seeds': [42, 123],
        'epochs': 100,
        'priority': 5,  # EXPERIMENTAL
        'time_estimate': 3.0
    }
}

class MultiSeedRunner:
    def __init__(self, output_dir='results/multiseed'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / 'execution_log.jsonl'
        
    def log_event(self, event_type, experiment, seed, data=None):
        """Log experiment events for tracking"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'experiment': experiment,
            'seed': seed,
            'data': data or {}
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def run_single_experiment(self, name, config, seed):
        """Run a single experiment with given seed"""
        print(f"\n{'='*60}")
        print(f"[{name}] Seed {seed} - Priority {config['priority']}")
        print(f"Estimated time: {config['time_estimate']} hours")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Build command
        cmd = [
            'python', f"src/{config['script']}", 
            '--seed', str(seed),
            '--epochs', str(config['epochs']),
            '--output-dir', f"results/multiseed/{name}/seed_{seed}"
        ]
        
        self.log_event('start', name, seed)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            elapsed = (time.time() - start_time) / 3600  # hours
            
            self.log_event('complete', name, seed, {
                'elapsed_hours': elapsed,
                'return_code': 0
            })
            
            print(f"✓ [{name}] Seed {seed} complete ({elapsed:.2f} hours)")
            return True
            
        except subprocess.CalledProcessError as e:
            elapsed = (time.time() - start_time) / 3600
            
            self.log_event('failed', name, seed, {
                'elapsed_hours': elapsed,
                'return_code': e.returncode,
                'error': str(e)
            })
            
            print(f"✗ [{name}] Seed {seed} FAILED ({elapsed:.2f} hours)")
            print(f"Error: {e.stderr}")
            return False
    
    def run_experiment_batch(self, name, config):
        """Run all seeds for a single experiment"""
        print(f"\n{'#'*60}")
        print(f"# Starting {name.upper()} ({len(config['seeds'])} seeds)")
        print(f"# Estimated total: {len(config['seeds']) * config['time_estimate']:.1f} hours")
        print(f"{'#'*60}\n")
        
        results = []
        for seed in config['seeds']:
            success = self.run_single_experiment(name, config, seed)
            results.append(success)
        
        # Summary
        success_count = sum(results)
        print(f"\n[{name}] Complete: {success_count}/{len(results)} successful")
        
        return results
    
    def run_all(self, experiments=None):
        """Run all experiments or specified subset"""
        if experiments is None:
            experiments = list(EXPERIMENTS.keys())
        
        # Sort by priority
        sorted_exps = sorted(
            [(name, EXPERIMENTS[name]) for name in experiments],
            key=lambda x: x[1]['priority']
        )
        
        total_time = 0
        for name, config in sorted_exps:
            total_time += len(config['seeds']) * config['time_estimate']
        
        print(f"\n{'*'*60}")
        print(f"* 60-HOUR MULTI-SEED VALIDATION PLAN")
        print(f"* Total experiments: {len(sorted_exps)}")
        print(f"* Estimated time: {total_time:.1f} hours")
        print(f"* Start: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'*'*60}\n")
        
        # Execute in priority order
        for name, config in sorted_exps:
            self.run_experiment_batch(name, config)
        
        print(f"\n{'*'*60}")
        print(f"* ALL EXPERIMENTS COMPLETE")
        print(f"* Elapsed: {total_time:.1f} hours")
        print(f"{'*'*60}\n")
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate execution summary"""
        events = []
        with open(self.log_file, 'r') as f:
            events = [json.loads(line) for line in f]
        
        summary = {
            'total_experiments': len(set(e['experiment'] for e in events)),
            'total_seeds': len([e for e in events if e['type'] == 'start']),
            'successful': len([e for e in events if e['type'] == 'complete']),
            'failed': len([e for e in events if e['type'] == 'failed']),
            'total_time_hours': sum(
                e['data'].get('elapsed_hours', 0) 
                for e in events 
                if e['type'] in ['complete', 'failed']
            )
        }
        
        summary_file = self.output_dir / 'execution_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n Summary saved to: {summary_file}")
        print(json.dumps(summary, indent=2))

def main():
    parser = argparse.ArgumentParser(description='Multi-seed validation runner')
    parser.add_argument('--experiments', nargs='+', 
                        choices=list(EXPERIMENTS.keys()),
                        help='Specific experiments to run (default: all)')
    parser.add_argument('--test-mode', action='store_true',
                        help='Test mode: only run first seed of each experiment')
    
    args = parser.parse_args()
    
    # Test mode: reduce seeds for quick validation
    if args.test_mode:
        print("\n⚠️  TEST MODE: Running only first seed of each experiment\n")
        for name in EXPERIMENTS:
            EXPERIMENTS[name]['seeds'] = EXPERIMENTS[name]['seeds'][:1]
    
    runner = MultiSeedRunner()
    runner.run_all(args.experiments)

if __name__ == '__main__':
    main()
