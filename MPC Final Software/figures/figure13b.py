"""
Generate Figure 13b
"""
import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

from simopt.experiment_base import ProblemSolver, plot_area_scatterplots, post_normalize, plot_progress_curves, plot_solvability_cdfs, read_experiment_results, plot_solvability_profiles, plot_terminal_scatterplots, plot_terminal_progress


# Problems factors used in experiments
# SSCONT
demand_means = [200.0]
lead_means = [3.0]

# RUNNING AND POST-PROCESSING EXPERIMENTS
L = 300
solvers = ["ASTROBFDF_coef=2.0", "ASTRODF_coef=2.0"] 
problems = ["SSCONTMF"]

experiments = []

# SSCONT
for dm in demand_means:
    for lm in lead_means:
        model_fixed_factors = {"demand_mean": dm,
                               "lead_mean": lm}

        # Temporarily store experiments on the same problem for post-normalization.
        experiments_same_problem = []
        solver_fixed_factors = {}
        for solver in solvers:
            solver_name = solver

            problem_rename = f"SSCONTMF-1_dm={dm}_lm={lm}"
            file_name = f"{solver}_on_{problem_rename}"
            # Load experiment.
            new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
            experiments_same_problem.append(new_experiment)

        # Post-normalize experiments with L.
        # Provide NO proxies for f(x0), f(x*), or f(x).
        post_normalize(experiments=experiments_same_problem, n_postreps_init_opt=L)

for solver in solvers:
    experiments_same_solver = []

    solver_display = solver
    if solver == "ASTROBFDF_coef=2.0":
        solver_display = "ASTRO-BFDF"
    elif solver == "ASTRODF_coef=2.0":
        solver_display = "ASTRO-DF"
        
    for problem in problems:
        if problem == "SSCONTMF":
            # Load SSCONT .pickle files
            for dm in demand_means:
                for lm in lead_means:
                    problem_rename = f"{problem}-1_dm={dm}_lm={lm}"
                    file_name = f"{solver}_on_{problem_rename}"
                    # Load experiment.
                    new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
                    # Rename problem to produce nicer plot labels.
                    new_experiment.problem.name = fr"{problem}-1 with $\mu_D={round(dm)}$ and $\mu_L={round(lm)}$"
                    new_experiment.solver.name = solver_display
                    experiments_same_solver.append(new_experiment)

    experiments.append(experiments_same_solver)

# PLOTTING
n_solvers = len(experiments)
n_problems = len(experiments[0])

CI_param = True
alpha = 0.01

for i in range(n_problems):
    plot_progress_curves([experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type="mean", all_in_one=True, plot_CIs=CI_param, print_max_hw=False, normalize=False)
