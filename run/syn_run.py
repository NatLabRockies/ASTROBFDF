"""
Generate .pickle fils in /experiments/outputs (synthetic problems)
"""
import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))) # type:ignore

from simopt.experiment_base import ProblemSolver, plot_area_scatterplots, post_normalize, plot_progress_curves, plot_solvability_cdfs, read_experiment_results, plot_solvability_profiles, plot_terminal_scatterplots, plot_terminal_progress

def main():
    hf_sigma2 = [10,15]
    lf_sigma2 = [5,10,15]
    
    #hf_sigma2 = [5]
    #lf_sigma2 = [15]

    correlation = [0.1, 0.5, 0.9]
    
    # RUNNING AND POST-PROCESSING EXPERIMENTS
    M = 20
    N = 100
    L = 200

    solvers = ["ASTROBFDF_coef=1.5", "ASTROBFDF_coef=2.0", "ASTROBFDF_coef=0.5", "ASTROBFDF_coef=1",
               "ASTRODF_coef=1.5", "ASTRODF_coef=2.0", "ASTRODF_coef=0.5", "ASTRODF_coef=1",
               "BFSAG_lr=1_n=10", "BFSAG_lr=0.5_n=10", "BFSAG_lr=2.0_n=10", "BFSAG_lr=0.1_n=10", "BFSAG_lr=0.01_n=10",
               "BFSAG_lr=0.01_n=20", "BFSAG_lr=0.01_n=30",
               "ADAM_lr=1_n=10", "ADAM_lr=0.5_n=10", "ADAM_lr=0.1_n=10", "ADAM_lr=2.0_n=10", "ADAM_lr=0.01_n=10",
               "ADAM_lr=1_n=20", "ADAM_lr=1_n=30",
               "BFBO_c=0.1_n=30", "BFBO_c=0.1_n=20", "BFBO_c=0.1_n=10", "BFBO_c=1_n=10",
               "ASTROBFDF_c=1_coef=2.0", "BFSAG_c=1_lr=0.01_n=10",
               ]
    
    #problems = ["FORRETAL", "BRANIN", "COLVILLE", "ROSENBROCK"]
    problems = ["ROSENBROCK"]

    # SYN
    for problem in problems:
        for hf_sig2 in hf_sigma2:
            for sig2 in lf_sigma2:
                for cor in correlation:
                    model_fixed_factors = {"sigma2": sig2,
                                        "sigma1": hf_sig2,
                                            "cor": cor}
                    if problem == "ROSENBROCK":
                        problem_fixed_factors = {"budget": 2000}
                    elif problem == "COLVILLE":
                        problem_fixed_factors = {"budget": 1000}
                    else:
                        problem_fixed_factors = {}
                    problem_rename = f"{problem}-1_hf_sig2={hf_sig2}_lf_sig2={sig2}_cor={cor}"

                    # Temporarily store experiments on the same problem for post-normalization.
                    experiments_same_problem = []
                    solver_fixed_factors = {}
                    for solver in solvers:
                        solver_name = solver

                        if solver == "BFSAG_lr=1_n=10":
                            solver_name = "BFSAG"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "lr": 1, "N":50, "Nl":20, "Nh":10}
                        elif solver == "BFSAG_lr=0.5_n=10":
                            solver_name = "BFSAG"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "lr": 0.5, "N":50, "Nl":20, "Nh":10}
                        elif solver == "BFSAG_lr=2.0_n=10":
                            solver_name = "BFSAG"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "lr": 2.0, "N":50, "Nl":20, "Nh":10}
                        elif solver == "BFSAG_lr=0.1_n=10":
                            solver_name = "BFSAG"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "lr": 0.1, "N":50, "Nl":20, "Nh":10}
                        elif solver == "BFSAG_lr=0.01_n=10":
                            solver_name = "BFSAG"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "lr": 0.01, "N":50, "Nl":20, "Nh":10}

                        elif solver == "BFSAG_lr=0.01_n=5":
                            solver_name = "BFSAG"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "lr": 0.01, "N":50, "Nl":10, "Nh":5}
                        elif solver == "BFSAG_lr=0.01_n=10":
                            solver_name = "BFSAG"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "lr": 0.01, "N":50, "Nl":20, "Nh":10}
                        elif solver == "BFSAG_lr=0.01_n=20":
                            solver_name = "BFSAG"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "lr": 0.01, "N":100, "Nl":40, "Nh":20}
                        elif solver == "BFSAG_lr=0.01_n=30":
                            solver_name = "BFSAG"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "lr": 0.01, "N":100, "Nl":60, "Nh":30}
                        elif solver == "BFSAG_c=1_lr=0.01_n=10":
                            solver_name = "BFSAG"
                            solver_fixed_factors = {"costs_mf": [1, 1], "lr": 0.01, "N":50, "Nl":20, "Nh":10}
                    
                    # ADAM
                        if solver == "ADAM_lr=1_n=10":
                            solver_name = "ADAM"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "lr": 1, "r":10}
                        elif solver == "ADAM_lr=0.5_n=10":
                            solver_name = "ADAM"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "lr": 0.5, "r":10}
                        elif solver == "ADAM_lr=0.1_n=10":
                            solver_name = "ADAM"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "lr": 0.1, "r":10}
                        elif solver == "ADAM_lr=0.01_n=10":
                            solver_name = "ADAM"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "lr": 0.01, "r":10}
                        elif solver == "ADAM_lr=2.0_n=10":
                            solver_name = "ADAM"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "lr": 2.0, "r":10}

                        if solver == "ADAM_lr=1_n=20":
                            solver_name = "ADAM"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "lr": 1, "r":20}
                        elif solver == "ADAM_lr=1_n=30":
                            solver_name = "ADAM"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "lr": 1, "r":30}

                            
                        if solver == "ASTROBFDF_coef=1":
                            solver_name = "ASTROBFDF"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "coef_delta_initial": 1}
                        elif solver == "ASTROBFDF_coef=0.5":
                            solver_name = "ASTROBFDF"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "coef_delta_initial": 0.5}
                        elif solver == "ASTROBFDF_coef=0.1":
                            solver_name = "ASTROBFDF"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "coef_delta_initial": 0.1}
                        elif solver == "ASTROBFDF_coef=1.5":
                            solver_name = "ASTROBFDF"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "coef_delta_initial": 1.5}
                        elif solver == "ASTROBFDF_coef=2.0":
                            solver_name = "ASTROBFDF"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "coef_delta_initial": 2.0}
                        elif solver == "ASTROBFDF_c=1_coef=2.0":
                            solver_name = "ASTROBFDF"
                            solver_fixed_factors = {"costs_mf": [1, 1], "coef_delta_initial": 2.0}

                            
                        if solver == "ASTRODF_coef=1":
                            solver_name = "ASTRODF"
                            solver_fixed_factors = {"coef_delta_initial": 1}
                        elif solver == "ASTRODF_coef=0.5":
                            solver_name = "ASTRODF"
                            solver_fixed_factors = {"coef_delta_initial": 0.5}
                        elif solver == "ASTRODF_coef=1.5":
                            solver_name = "ASTRODF"
                            solver_fixed_factors = {"coef_delta_initial": 1.5}
                        elif solver == "ASTRODF_coef=2.0":
                            solver_name = "ASTRODF"
                            solver_fixed_factors = {"coef_delta_initial": 2.0}


                        elif solver == "BFBO_c=0.1_n=30":
                            solver_name = "BFBO"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "r": 30}
                        elif solver == "BFBO_c=1_n=30":
                            solver_name = "BFBO"
                            solver_fixed_factors = {"costs_mf": [1, 1], "r": 30}
                        elif solver == "BFBO_c=0.1_n=20":
                            solver_name = "BFBO"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "r": 20}
                        elif solver == "BFBO_c=0.1_n=10":
                            solver_name = "BFBO"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "r": 10}
                        elif solver == "BFBO_c=1_n=10":
                            solver_name = "BFBO"
                            solver_fixed_factors = {"costs_mf": [1, 1], "r": 10}
                        
                        

                        # Loop over solvers:
                        new_experiment = ProblemSolver(solver_name=solver_name,
                                                        solver_rename=solver,
                                                        solver_fixed_factors=solver_fixed_factors,
                                                        problem_name=problem+"-1",
                                                        problem_rename=problem_rename,
                                                        problem_fixed_factors=problem_fixed_factors,
                                                        model_fixed_factors=model_fixed_factors
                                                        )
                        # Run experiment with M.
                        new_experiment.run(n_macroreps=M)
                        # Post replicate experiment with N.
                        new_experiment.post_replicate(n_postreps=N)
                        experiments_same_problem.append(new_experiment)

                    # Post-normalize experiments with L.
                    # Provide NO proxies for f(x0), f(x*), or f(x).
                    post_normalize(experiments=experiments_same_problem, n_postreps_init_opt=L)


if (__name__ == "__main__"):
    main()