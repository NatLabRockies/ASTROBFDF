"""
Generate .pickle fils in /experiments/outputs (FORRETAL)
"""
import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))) # type:ignore

from simopt.experiment_base import ProblemSolver, plot_area_scatterplots, post_normalize, plot_progress_curves, plot_solvability_cdfs, read_experiment_results, plot_solvability_profiles, plot_terminal_scatterplots, plot_terminal_progress

def main():
    hf_sigma2 = [15]
    lf_sigma2 = [15]
    correlation = [0.1, 0.9]
        
    # RUNNING AND POST-PROCESSING EXPERIMENTS
    M = 20
    N = 100
    L = 200
    
    solvers = ["ASTROBFDF", "ASTRODF", "BFBO_c=0.1_n=10", "BFBO_c=0.1_n=20"]
    problems = ["FORRETAL"]

    # SYN
    for problem in problems:
        for hf_sig2 in hf_sigma2:
            for sig2 in lf_sigma2:
                for cor in correlation:
                    model_fixed_factors = {"sigma2": sig2,
                                        "sigma1": hf_sig2,
                                            "cor": cor}
                    problem_fixed_factors = {}
                    problem_rename = f"{problem}-1_hf_sig2={hf_sig2}_lf_sig2={sig2}_cor={cor}"

                    # Temporarily store experiments on the same problem for post-normalization.
                    experiments_same_problem = []
                    solver_fixed_factors = {}
                    for solver in solvers:
                        solver_name = solver
                        if solver == "ASTROMFDF":
                            solver_fixed_factors = {"costs_mf": [1, 0.1]}
                        elif solver == "BFBO_c=0.1_n=20":
                            solver_name = "BFBO"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "r": 20}
                        elif solver == "BFBO_c=0.1_n=10":
                            solver_name = "BFBO"
                            solver_fixed_factors = {"costs_mf": [1, 0.1], "r": 10}
                        

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