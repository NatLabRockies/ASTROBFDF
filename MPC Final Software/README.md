## Description

This repository aims to demonstrate the effect of adaptive sampling-based bi-fidelity stochastic trust region method (ASTRO-BFDF). ASTRO-BFDF, derived from a derivative-free adaptive sampling trust-region optimization (ASTRO-DF) ([Shashaani et al. 2018](https://epubs.siam.org/doi/abs/10.1137/15M1042425), [Ha and Shashaani 2023](https://ieeexplore.ieee.org/abstract/document/10408143)), intended to efficiently solve the bi-fidelity simulation optimization.

## Building

Tested with **Python 3.12.1**.

Create and activate an environment with Python 3.12.1, then install the packages listed in requirements.txt.

```
conda create -n astrobfdf python=3.12.1 -y
conda activate astrobfdf
pip install -r requirements.txt
```

## Results

All detailed results are available in [plots](experiments/plots) folder.

## Replicating

To replicate the results presented in the paper, run the code located in the [figures](figures) folder. This will generate the corresponding plots found in the [plots](experiments/plots) folder using the output .pickle files from the [outputs](experiments/outputs) folder. For instance, to generate Figure 6, execute the script
```
python figures/figure6.py
```

To replicate these .pickle files, execute the code in the [run](run) folder. For example, to generate the output .pickle files regarding discrete event simulation, execute the script.

```
python run/des_run.py
```

## Usage Notes for Synthetic Experiments

To avoid discrepancies between synthetic and real-world scenarios, **keep the simulation budget moderate** when running the provided synthetic experiments. In synthetic problems, simulation outputs are generated almost instantaneously. If you set an excessively large budget, the optimization overhead will dominate the runtime, which does not reflect the intended assumptions.

**Recommendation:** Do **not** set excessively large budgets for synthetic experiments.



## Example Script: `example_test.py`

We have added an example script to help users apply **ASTRO-BFDF** to their own problems without deep knowledge of the SimOpt framework.

### What the script demonstrates:
- How to define custom **high- and low-fidelity functions**, noise rules, and cost structure.
- How to configure **problem settings** (e.g., initial solution, budget) and solver parameters (e.g., batch sizes).
- How to run **ASTRO-BFDF** (and other single-fidelity solvers) on a single bi-fidelity optimization problem.
- How to **post-process results and generate plots** (progress curves and normalized comparisons).

### Hierarchical Model Specifications & Robust Fallback
To ensure the system remains stable and "simulatable" under minimal configuration, we utilize a hierarchical specification system:
- **Default Handling:** Every core model (e.g., [`example.py`](simopt/models/example.py)) contains a `specifications` dictionary with pre-defined default values for functions, noise rules, and costs.
- **Robust Fallback:** If optional parameters are omitted in this testing interface, the system will **not** trigger an error. Instead, it automatically falls back to the internal defaults defined in the core model.
- **Consistency:** If your custom input is mathematically identical to the internal default, the numerical output will remain consistent regardless of whether the parameter is explicitly provided.

### Batch Sizes for Adaptive Sampling (`batch_sizes = [b1, b2]`)

Users may observe apparent "non-termination" with ASTRO-BFDF when the budget is set very large. This is typically not a true failure. Near an optimal solution, ASTRO-BFDF requires increasingly accurate function estimates, so the adaptive sampling rule performs many internal updates and stopping checks. In fast synthetic settings (where simulation itself is nearly instantaneous), this overhead can dominate runtime.

To improve efficiency, the example interface now supports user-defined batch sizes for HF/LF sampling:
```
batch_sizes = [b1, b2]
```
- `b1`: number of high-fidelity samples collected per update
- `b2`: number of low-fidelity samples collected per update

Larger batch sizes reduce how frequently the solver performs stopping-rule checks, which can significantly reduce overhead in high-precision, near-optimal phases.

Example:
```
budget = 2000
costs = [0.1, 0.01]
batch_sizes = [10, 20]
```
In our internal tests for this setting, the solver finishes in under 15 seconds, while the previous unit-batch setting could appear to stall due to frequent internal checks.

### Run the example:
```
python example_test.py
```

### Outputs:
- Experiment results are stored as .pickle files in the experiments/outputs/ folder.
- Performance plots (e.g., progress curves, solvability profiles) are displayed automatically.
- Normalized comparison plots are available in the experiments/plots/ folder after running post-processing.
- Intermediate statistics and solver performance metrics are also saved for reproducibility.
- During execution, detailed iteration logs are printed in the console, including:
  - Iteration number
  - Current solution (`new_x`)
  - Trust-region radius (`delta`)
  - Correlation parameter
  - Expended simulation budget
  
#### Example Log:
```
    Iter   new_x                delta                correlation parameter      Budget Expended
    ==========================================================================================
    1      (10.0,)              [0.5000, 0.5000]            0.9123                     10.00
    2      (8.5,)               [0.2500, 0.2000]            0.9345                     20.00
```
This script focuses on running a single optimization problem for clarity and is easily customizable. Detailed comments are provided inside `example_test.py` for user reference. 


## Acknowledgement

This software framework is built upon the foundation provided by [SimOpt](https://github.com/simopt-admin/simopt). We acknowledge the contributions of D.J. Eckman, S.G. Henderson, S. Shashaani, and R. Pasupathy for their work on SimOpt, which has greatly facilitated the development of this project.
