"""
Generate .pickle fils in /experiments/outputs (discrete event simulation)
"""
import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))) # type:ignore

from simopt.experiment_base import ProblemSolver, post_normalize 

def main():
    # Problems factors used in experiments
    # MM1
    mu_set = [1.0]
    lambda_setup = [1.0,2.0,3.0,4.0,5.0]

    # SSCONT
    demand_means = [25.0, 50.0, 100.0, 200.0, 400.0]
    lead_means = [1.0, 3.0, 6.0, 9.0]


    # RUNNING AND POST-PROCESSING EXPERIMENTS
    M = 20
    N = 200
    L = 200         


    # Five solvers.
    solvers = ["ASTROBFDF_coef=0.5", "ASTROBFDF_coef=1", "ASTROBFDF_coef=1.5", "ASTROBFDF_coef=2.0",
               "ASTRODF_coef=0.5", "ASTRODF_coef=1", "ASTRODF_coef=1.5", "ASTRODF_coef=2.0",
               "ADAM_coef=0.5_n=10", "ADAM_coef=1_n=10", "ADAM_coef=1.5_n=10", "ADAM_coef=2.0_n=10",
               "ADAM_coef=2.0_n=20","ADAM_coef=2.0_n=30",
               "BFSAG_coef=0.5_n=10", "BFSAG_coef=1_n=10", "BFSAG_coef=1.5_n=10", "BFSAG_coef=2.0_n=10",
               "BFSAG_coef=2.0_n=20","BFSAG_coef=2.0_n=30",
               "BFBO_n=10", "BFBO_n=20", "BFBO_n=30"
               ]

    # MM1
    for mu in mu_set:
        for ld in lambda_setup:
            model_fixed_factors = {"lambda": ld,
                                    "mu": mu}
            problem_fixed_factors = {"budget": 200}
            problem_rename = f"MM1MF-1_mu={mu}_ld={ld}"

            # Temporarily store experiments on the same problem for post-normalization.
            experiments_same_problem = []
            solver_fixed_factors = {}
            for solver in solvers:
                solver_name = solver
                
                if solver == "ASTROBFDF_coef=1":
                    solver_name = "ASTROBFDF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 1}
                elif solver == "ASTROBFDF_coef=0.5":
                    solver_name = "ASTROBFDF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 0.5}
                elif solver == "ASTROBFDF_coef=0.1":
                    solver_name = "ASTROBFDF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 0.1}
                elif solver == "ASTROBFDF_coef=1.5":
                    solver_name = "ASTROBFDF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 1.5}
                elif solver == "ASTROBFDF_coef=2.0":
                    solver_name = "ASTROBFDF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 2.0}
                    
                if solver == "ASTRODF_coef=1":
                    solver_name = "ASTRODF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 1}
                elif solver == "ASTRODF_coef=0.5":
                    solver_name = "ASTRODF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 0.5}
                elif solver == "ASTRODF_coef=1.5":
                    solver_name = "ASTRODF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 1.5}
                elif solver == "ASTRODF_coef=2.0":
                    solver_name = "ASTRODF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 2.0}
                elif solver == "ASTRODF_coef=0.1":
                    solver_name = "ASTRODF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 0.1}

                if solver == "ADAM_coef=1_n=10":
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 1, "r":10}
                elif solver == "ADAM_coef=0.1_n=10":
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 0.1, "r":10}
                elif solver == "ADAM_coef=0.5_n=10":    
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 0.5, "r":10}
                elif solver == "ADAM_coef=1.5_n=10":
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 1.5, "r":10}
                elif solver == "ADAM_coef=2.0_n=10":
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 2.0, "r":10}
                elif solver == "ADAM_coef=2.0_n=20":
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 2.0, "r":20}
                elif solver == "ADAM_coef=2.0_n=30":
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 2.0, "r":30}

                if solver == "BFSAG_coef=1_n=10":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 1, "N":50, "Nl":20, "Nh":10}
                elif solver == "BFSAG_coef=0.1_n=10":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 0.1, "N":50, "Nl":20, "Nh":10}
                elif solver == "BFSAG_coef=0.5_n=10":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 0.5, "N":50, "Nl":20, "Nh":10}
                elif solver == "BFSAG_coef=1.5_n=10":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 1.5, "N":50, "Nl":20, "Nh":10}
                elif solver == "BFSAG_coef=2.0_n=10":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 2.0, "N":50, "Nl":20, "Nh":10}
                elif solver == "BFSAG_coef=2.0_n=20":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 2.0, "N":100, "Nl":40, "Nh":20}
                elif solver == "BFSAG_coef=2.0_n=30":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 2.0, "N":100, "Nl":60, "Nh":30}

                if solver == "BFBO_n=20":
                    solver_name = "BFBO"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "r": 20}
                elif solver == "BFBO_n=10":
                    solver_name = "BFBO"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "r": 10}
                elif solver == "BFBO_n=30":
                    solver_name = "BFBO"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "r": 30}

                # Loop over solvers:
                new_experiment = ProblemSolver(solver_name=solver_name,
                                                solver_rename=solver,
                                                solver_fixed_factors=solver_fixed_factors,
                                                problem_name="MM1MF-1",
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

    # Second problem: SSCONT
    for dm in demand_means:
        for lm in lead_means:
            model_fixed_factors = {"demand_mean": dm,
                                "lead_mean": lm}
            # Default budget for (s,S) inventory problem = 1000 replications.
            # RS with sample size of 100 will get through only 10 iterations.
            problem_fixed_factors = {"budget": 1000}
            problem_rename = f"SSCONTMF-1_dm={dm}_lm={lm}"

            # Temporarily store experiments on the same problem for post-normalization.
            experiments_same_problem = []
            solver_fixed_factors = {}
            for solver in solvers:
                solver_name = solver

                if solver == "BFSAG_lr=1_n=10":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "lr": 1, "N":50, "Nl":20, "Nh":10}
                elif solver == "BFSAG_lr=0.5_n=10":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "lr": 0.5, "N":50, "Nl":20, "Nh":10}
                elif solver == "BFSAG_lr=2.0_n=10":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "lr": 2.0, "N":50, "Nl":20, "Nh":10}
                elif solver == "BFSAG_lr=1.5_n=10":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "lr": 1.5, "N":50, "Nl":20, "Nh":10}
                elif solver == "BFSAG_lr=0.1_n=10":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "lr": 0.1, "N":50, "Nl":20, "Nh":10}
                elif solver == "BFSAG_lr=0.01_n=10":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "lr": 0.01, "N":50, "Nl":20, "Nh":10}
            
            # ADAM
                if solver == "ADAM_lr=1_n=10":
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "lr": 1, "r":10}
                elif solver == "ADAM_lr=0.5_n=10":
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "lr": 0.5, "r":10}
                elif solver == "ADAM_lr=1.5_n=10":
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "lr": 1.5, "r":10}
                elif solver == "ADAM_lr=0.1_n=10":
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "lr": 0.1, "r":10}
                elif solver == "ADAM_lr=0.01_n=10":
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "lr": 0.01, "r":10}
                elif solver == "ADAM_lr=2.0_n=10":
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "lr": 2, "r":10}
                
                
                if solver == "ASTROBFDF_coef=1":
                    solver_name = "ASTROBFDF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 1}
                elif solver == "ASTROBFDF_coef=0.5":
                    solver_name = "ASTROBFDF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 0.5}
                elif solver == "ASTROBFDF_coef=0.1":
                    solver_name = "ASTROBFDF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 0.1}
                elif solver == "ASTROBFDF_coef=1.5":
                    solver_name = "ASTROBFDF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 1.5}
                elif solver == "ASTROBFDF_coef=2.0":
                    solver_name = "ASTROBFDF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 2.0}
                    
                    
                if solver == "ASTRODF_coef=1":
                    solver_name = "ASTRODF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 1}
                elif solver == "ASTRODF_coef=0.5":
                    solver_name = "ASTRODF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 0.5}
                elif solver == "ASTRODF_coef=1.5":
                    solver_name = "ASTRODF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 1.5}
                elif solver == "ASTRODF_coef=2.0":
                    solver_name = "ASTRODF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 2.0}
                elif solver == "ASTRODF_coef=0.1":
                    solver_name = "ASTRODF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 0.1}

                if solver == "ADAM_coef=1_n=10":
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 1, "r":10}
                elif solver == "ADAM_coef=0.1_n=10":
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 0.1, "r":10}
                elif solver == "ADAM_coef=0.5_n=10":    
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 0.5, "r":10}
                elif solver == "ADAM_coef=1.5_n=10":
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 1.5, "r":10}
                elif solver == "ADAM_coef=2.0_n=10":
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 2.0, "r":10}
                elif solver == "ADAM_coef=2.0_n=20":
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 2.0, "r":20}
                elif solver == "ADAM_coef=2.0_n=30":
                    solver_name = "ADAM"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 2.0, "r":30}

                if solver == "BFSAG_coef=1_n=10":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 1, "N":50, "Nl":20, "Nh":10}
                elif solver == "BFSAG_coef=0.1_n=10":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 0.1, "N":50, "Nl":20, "Nh":10}
                elif solver == "BFSAG_coef=0.5_n=10":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 0.5, "N":50, "Nl":20, "Nh":10}
                elif solver == "BFSAG_coef=1.5_n=10":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 1.5, "N":50, "Nl":20, "Nh":10}
                elif solver == "BFSAG_coef=2.0_n=10":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 2.0, "N":50, "Nl":20, "Nh":10}
                elif solver == "BFSAG_coef=2.0_n=20":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 2.0, "N":100, "Nl":40, "Nh":20}
                elif solver == "BFSAG_coef=2.0_n=30":
                    solver_name = "BFSAG"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coefficient": 2.0, "N":100, "Nl":60, "Nh":30}
                    
                if solver == "BFBO_n=20":
                    solver_name = "BFBO"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "r": 20}
                elif solver == "BFBO_n=10":
                    solver_name = "BFBO"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "r": 10}
                elif solver == "BFBO_n=30":
                    solver_name = "BFBO"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "r": 30}

            
                # Loop over solvers:
                new_experiment = ProblemSolver(solver_name=solver_name,
                                            solver_rename=solver,
                                            solver_fixed_factors=solver_fixed_factors,
                                            problem_name="SSCONTMF-1",
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