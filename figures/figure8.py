"""
Generate Figure 8
"""
import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

from simopt.experiment_base import ProblemSolver, plot_area_scatterplots, post_normalize, plot_progress_curves, plot_solvability_cdfs, read_experiment_results, plot_solvability_profiles, plot_terminal_scatterplots, plot_terminal_progress


# Problems factors used in experiments
# MM1
mu_set = [1.0]
lambda_setup = [1.0,2.0,3.0,4.0,5.0]


# SSCONT
demand_means = [25.0, 50.0, 100.0, 200.0, 400.0]
lead_means = [1.0, 3.0, 6.0, 9.0]

# RUNNING AND POST-PROCESSING EXPERIMENTS
L = 200
solvers = ["ASTROBFDF_coef=2.0", "ASTRODF_coef=2.0", "ADAM_coef=2.0_n=10", "BFSAG_coef=2.0_n=10", "BFBO_n=10"] 
problems = ["MM1MF", "SSCONTMF"]

experiments = []


# MM1
for mu in mu_set:
    for ld in lambda_setup:
        model_fixed_factors = {"lambda": ld,
                                "mu": mu}

        # Temporarily store experiments on the same problem for post-normalization.
        experiments_same_problem = []
        solver_fixed_factors = {}
        for solver in solvers:
            solver_name = solver

            problem_rename = f"MM1MF-1_mu={mu}_ld={ld}"
            file_name = f"{solver}_on_{problem_rename}"
            # Load experiment.
            new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
            experiments_same_problem.append(new_experiment)

        # Post-normalize experiments with L.
        # Provide NO proxies for f(x0), f(x*), or f(x).
        post_normalize(experiments=experiments_same_problem, n_postreps_init_opt=L)

# Second problem: SSCONT
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
    elif solver == "ADAM_coef=2.0_n=10":
        solver_display = "ADAM"
    elif solver == "BFSAG_coef=2.0_n=10":
        solver_display = "BFSAG"
    elif solver == "BFBO_n=10":
        solver_display = "BFBO"

    for problem in problems:
        if problem == "MM1MF":
            # Load IRONORECONT .pickle files
            for mu in mu_set:
                for ld in lambda_setup:
                    problem_rename = f"{problem}-1_mu={mu}_ld={ld}"
                    file_name = f"{solver}_on_{problem_rename}"
                    # Load experiment.
                    new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
                    # Rename problem to produce nicer plot labels.
                    new_experiment.problem.name = fr"{problem}-1 with $mu={mu}$ and lambda={ld}"
                    new_experiment.solver.name = solver_display
                    experiments_same_solver.append(new_experiment)

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

plot_solvability_profiles(experiments, plot_type="cdf_solvability", solve_tol=alpha, all_in_one=True, plot_CIs=CI_param, print_max_hw=False)
