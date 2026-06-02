# MEVO: Meta-Evolutionary Optimizer with Surrogate-Based Optimization

## Overview

MEVO is an advanced optimization algorithm for machining parameter optimization that combines:

- **Gaussian Process / Artificial Neural Network** surrogate models for function approximation
- **MEPSO** (Micro Evolutionary Particle Swarm Optimizer) for acquisition function optimization
- **Evolutionary Operators** (crossover and mutation) for generating candidate solutions
- **Active Learning Strategies** with adaptive search space reduction
- **Multi-Objective Desirability Functions** for simultaneous optimization of competing objectives

The algorithm iteratively samples new points by optimizing an acquisition function, updates the surrogate model with actual evaluations, and maintains an archive of evaluated solutions to avoid redundant evaluations.

## Reference

If you use MEVO in your research, please cite:

```
Velázquez-López, A., Batres, R., Miranda-Valenzuela, J. C., & Calderón-Nájera, J. D. (2026). 
Experimental assessment of the metamodel-based evolutionary optimizer (MEVO) for machining optimization. 
International Journal of Advanced Manufacturing Technology
```

## Installation

### Requirements

- Python 3.7 or higher
- Dependencies listed in `requirements.txt`

### Setup

#### Local Installation (Windows/Linux/macOS)

1. Clone or download this repository:
```bash
cd /path/to/mevo
```

2. Create a virtual environment (recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

#### Google Colab Installation

The code is compatible with Google Colab and will automatically detect the environment:

```python
# Upload files to Colab
from google.colab import files
files.upload()  # Upload mevo_machining.py, LHS_MEVO_4_Points.csv, and micropso.py

# Run directly - no additional setup needed
exec(open('mevo_machining.py').read())
```

## Input Data Format

### CSV File: Initial LHS Points

Create a CSV file (e.g., `LHS_MEVO_4_Points.csv`) with initial Latin Hypercube Sampling points:

```
index,ap,fz,Vc,SE,SR
0,1.2,0.040,60,10.5,1.5
1,1.5,0.050,75,12.3,1.8
2,1.3,0.045,65,11.2,1.6
3,1.4,0.048,70,11.8,1.7
```

**Columns:**
- `index`: Point number (0-based)
- `ap`: Depth of cut [mm]
- `fz`: Feed per tooth [mm/tooth]
- `Vc`: Cutting speed [m/min]
- `SE`: Surface Finish [response variable]
- `SR`: Tool Stress [response variable]

## Configuration

All configuration parameters are defined at the top of `mevo_machining.py`. Key parameters:

### 1. Evaluation Mode
```python
EVALUATION_MODE = 'manual'  # or 'automatic'
```
- `'manual'`: Prompts user to input observed SE and SR values after each evaluation
- `'automatic'`: Uses fitted regression models for prediction

### 2. Optimization Mode
```python
OPTIMIZATION_MODE = "DF_MOO"  # Options: "single_objective", "DF_SOO", "DF_MOO"
```

### 3. Multi-Objective Variables
```python
DF_MOO_VARIABLES = ["SE", "SR"]  # Variables to optimize simultaneously
```

### 4. Optimization Intentions
```python
OPTIMIZATION_INTENTIONS = {
    "SE": "min",  # Minimize surface finish
    "SR": "min",  # Minimize stress
    "MRR": "max"  # Maximize material removal rate
}
```

### 5. Desirability Ranges
```python
DESIRABILITY_RANGES = {
    "SE": (5.91011, 16.6839),           # (ideal, worst)
    "SR": (0.66150, 3.0100),            # (ideal, worst)
    "MRR": (3850.276383, 12994.682794)  # (worst, ideal)
}
```

### 6. Stopping Criterion
```python
MAX_EVALUATIONS = 16  # Maximum number of function evaluations
```

### 7. Evolutionary Operators
```python
PROB_CROSSOVER = 0.1   # Probability of crossover in each iteration
PROB_MUTATION = 0.2    # Probability of mutation in each iteration
```

For detailed documentation of all parameters, see the configuration section in `mevo_machining.py`.

## Usage

### Basic Execution

1. Ensure your initial LHS CSV file is in the working directory
2. Update configuration parameters as needed
3. Run the algorithm:

```bash
python mevo_machining.py
```

### Output Files

The algorithm generates several output files:

- `statistics_YYYYMMDD_HHMMSS_run1.csv`: Detailed evaluation history
  - Point coordinates, objectives, desirability scores
  - Surrogate model metrics (R², RMSE)

- `convergence_YYYYMMDD_HHMMSS_run1.csv`: Convergence tracking
  - Best solution found at each iteration

- `summary_of_all_runs.csv`: Summary of all optimization runs
  - Best solution, cost, computation time

### Resuming Interrupted Runs

If an optimization is interrupted, MEVO automatically saves a checkpoint and can resume:

```bash
# Run will automatically detect and resume from checkpoint
python mevo_machining.py
```

Checkpoints are saved in `./checkpoint/` directory (or Google Drive if on Colab).

## Algorithm Workflow

1. **Initialization**
   - Load initial LHS points from CSV
   - Evaluate all initial points
   - Train initial surrogate model (ANN/RF/SVM)

2. **Main Loop** (for each iteration until MAX_EVALUATIONS):
   - **PSO Acquisition**: Optimize acquisition function using MicroEPSO
   - **Crossover** (probabilistic): Generate offspring from elite solutions
   - **Mutation** (probabilistic): Perturb best solution found
   - **Model Update**: Retrain surrogate with new evaluations
   - **Convergence Tracking**: Record best solution progress

3. **Finalization**
   - Report best solution found
   - Plot convergence history
   - Save summary statistics

## Surrogate Model Options

The algorithm supports multiple surrogate models. To change, modify the model initialization:

```python
# Artificial Neural Network (default)
model = MLPRegressor(hidden_layer_sizes=(8,), activation='logistic', ...)

# Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=None)

# Support Vector Regression
model = SVR(kernel='rbf', C=1, epsilon=0.05, gamma='scale')
```

## Multi-Objective Optimization

MEVO uses Desirability Function (DF) approach for multi-objective optimization:

### Individual Desirability Function
```
For minimization:  d = (worst - value) / (worst - ideal)
For maximization:  d = (value - worst) / (ideal - worst)
Range: [0, 1] where 1 = ideal, 0 = worst
```

### Composite Desirability (DF-MOO)
```
d_composite = (d_SE^w_SE * d_SR^w_SR * d_MRR^w_MRR)^(1/sum_weights)
```

Weights allow you to prioritize certain objectives.

## Reproducibility

To ensure reproducible results:

```python
seed = 42  # Set at top of mevo_machining.py
random.seed(seed)
np.random.seed(seed)
```

All stochastic operations use this seed for deterministic behavior.

## Troubleshooting

### Issue: Python not found on Windows
**Solution:** Use full path or add Python to system PATH
```bash
C:\Python39\python.exe mevo_machining.py
```

### Issue: Module not found (pandas, sklearn, etc.)
**Solution:** Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: CSV file not found
**Solution:** Ensure LHS CSV file is in the same directory as the Python script

### Issue: Manual evaluation prompts don't appear
**Solution:** Ensure `EVALUATION_MODE = 'manual'` in configuration

### Issue: Checkpoint not loading
**Solution:** Check that checkpoint file path is accessible and not corrupted

## Contact & Support

For questions, issues, or contributions, please refer to the research paper or contact the authors.

## License

Please refer to the license file for usage rights and restrictions.

---

**Last Updated:** June 2, 2026  
**Version:** 1.0  
**Status:** Ready for publication
