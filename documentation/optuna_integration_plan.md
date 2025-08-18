# Lightweight Optuna Hyperparameter Optimization

## 🎯 Simple Objective

Add basic hyperparameter optimization to find better configs than manual tuning. Keep it simple and focused.

---

## 📋 Implementation Tasks

**Total Estimated Time: 4-6 hours**

- [ ] **Task 1: Basic Optuna Script (2-3 hours)**
  - Add `optuna` to requirements
  - Create `scripts/hyperopt.py` that reuses existing training pipeline
  - Define search space for key hyperparameters (5-8 params)
  - Test with 10-20 trials
  - *Files to create: `scripts/hyperopt.py`*

- [ ] **Task 2: Dual Storage System (1 hour)**
  - Use SQLite for Optuna study persistence (required by Optuna)
  - Export results to `results/hyperopt_results.json` for human readability
  - Store best config as `results/best_config.yaml`
  - Simple text summary of results
  - *Files to modify: `scripts/hyperopt.py`*

- [ ] **Task 3: Integration & Testing (1-2 hours)**
  - Test that exported config works with `train_with_config.py`
  - Run small study to verify everything works
  - Basic error handling for failed trials
  - *Files to modify: `scripts/hyperopt.py`*

## 📁 Dual Storage File Structure

```
results/
├── optuna_study.db            # SQLite database (Optuna's internal persistence)
├── hyperopt_results.json      # Exported trial data (human-readable)
├── best_config.yaml          # Optimal config ready for production
└── hyperopt_summary.txt      # Human-readable results summary
```

## 📋 Key Hyperparameters to Optimize

```python
# Core parameters to search (5-8 total)
hyperparameters = {
    'hidden_dim': [128, 256, 384, 512],
    'learning_rate': [0.0005, 0.001, 0.002],  
    'dropout': [0.2, 0.3, 0.4],
    'batch_size': [16, 24, 32],
    'scheduler': ['plateau', 'cosine', 'cosine_restarts']
}
```

## 📊 Simple Results Format

**hyperopt_results.json:**
```json
{
  "study_name": "basic_hyperopt",
  "best_score": 0.6247,
  "best_params": {
    "hidden_dim": 256,
    "learning_rate": 0.001,
    "dropout": 0.3,
    "batch_size": 32,
    "scheduler": "cosine"
  },
  "all_trials": [
    {"trial": 0, "score": 0.5234, "params": {...}},
    {"trial": 1, "score": 0.6247, "params": {...}}
  ]
}
```

**hyperopt_summary.txt:**
```
Hyperparameter Optimization Results
==================================
Best boundary F1: 0.6247
Best parameters:
  - hidden_dim: 256
  - learning_rate: 0.001
  - dropout: 0.3
  - batch_size: 32  
  - scheduler: cosine

Completed 20 trials in 16.7 hours
```

## 🧪 Basic Script Structure

```python
# scripts/hyperopt.py - Simple implementation sketch

import optuna
import json
from train_with_config import setup_data_loaders, setup_model_and_training
from segmodel.train import Trainer
from segmodel.utils import TrainingConfig

def objective(trial):
    """Simple objective function."""
    # Sample hyperparameters
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 384, 512])
    learning_rate = trial.suggest_float('learning_rate', 0.0005, 0.002)
    dropout = trial.suggest_float('dropout', 0.2, 0.4)
    batch_size = trial.suggest_categorical('batch_size', [16, 24, 32])
    scheduler = trial.suggest_categorical('scheduler', ['plateau', 'cosine'])
    
    # Create config with sampled parameters
    config = TrainingConfig(
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        dropout=dropout,
        batch_size=batch_size,
        scheduler=scheduler,
        # ... other fixed parameters from base config
    )
    
    # Run training (reuse existing pipeline)
    # ... setup and training code ...
    
    # Return target metric
    return final_metrics['boundary_f1']

def main():
    # Create study with SQLite persistence
    study = optuna.create_study(
        direction='maximize',
        storage='sqlite:///results/optuna_study.db',
        study_name='hyperopt_study'
    )
    
    # Optimize (Optuna handles SQLite internally)
    study.optimize(objective, n_trials=20)
    
    # Export human-readable results
    export_results_to_json(study)
    export_best_config(study.best_params)
    create_summary_report(study)

def export_results_to_json(study):
    """Export Optuna study results to JSON for easy reading."""
    results = {
        'best_score': study.best_value,
        'best_params': study.best_params,
        'all_trials': [
            {
                'trial': trial.number,
                'score': trial.value,
                'params': trial.params,
                'status': str(trial.state)
            }
            for trial in study.trials
        ]
    }
    
    with open('results/hyperopt_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
```

## ✅ Simple Success Criteria

- [ ] Script runs 10-20 trials without crashing
- [ ] SQLite database properly stores study progress (enables resuming)
- [ ] Exports `best_config.yaml` that works with existing training
- [ ] JSON export contains all trial data in human-readable format
- [ ] Finds parameters that perform better than current manual configs
- [ ] Takes reasonable compute time (1-2 days for full study)

## 🎯 Benefits of Dual Storage

**SQLite (Optuna's requirement):**
- ✅ Required for Optuna study persistence
- ✅ Enables resuming interrupted studies
- ✅ Handles concurrent trials (if needed later)
- ✅ Built-in optimization tracking

**JSON Export (Human-readable):**
- ✅ Easy to inspect and analyze manually
- ✅ Simple to backup and share
- ✅ Can be processed by other tools
- ✅ Clear format for understanding results

**Ready to implement!** Single script with dual storage approach.
