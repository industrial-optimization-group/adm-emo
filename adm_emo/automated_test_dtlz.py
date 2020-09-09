import numpy as np
import pandas as pd

import baseADM
from baseADM import *
import generatePreference as gp

from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors

from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII

from pymoo.factory import get_problem, get_reference_directions
import rmetric as rm
from sklearn.preprocessing import Normalizer

# from desdeo_tools.scalarization.ASF import PointMethodASF as asf

# from tqdm import tqdm


# problem_names = ["DTLZ2", "DTLZ3", "DTLZ4"]
problem_names = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4"]
# n_objs = np.asarray([3, 4, 5, 6, 7, 8, 9])
n_objs = np.asarray([3])
K = 10
n_vars = K + n_objs - 1

num_gen_per_iter = [100]

algorithms = ["iRVEA", "iNSGAIII"]
column_names = (
    ["problem", "num_obj", "iteration", "num_gens", "reference_point"]
    + [algorithm + "_R_IGD" for algorithm in algorithms]
    + [algorithm + "_R_HV" for algorithm in algorithms]
)

excess_columns = [
    "_R_IGD",
    "_R_HV",
]

data = pd.DataFrame(columns=column_names)
data_row = pd.DataFrame(columns=column_names, index=[1])

counter = 1
total_count = len(num_gen_per_iter) * len(n_objs) * len(problem_names)
for gen in num_gen_per_iter:
    for n_obj, n_var in zip(n_objs, n_vars):
        for problem_name in problem_names:
            print(f"Loop {counter} of {total_count}")
            counter += 1
            problem = test_problem_builder(
                name=problem_name, n_of_objectives=n_obj, n_of_variables=n_var
            )
            problem.ideal = np.asarray([0] * n_obj)
            problem.nadir = abs(np.random.normal(size=n_obj, scale=0.15)) + 1

            true_nadir = np.asarray([1] * n_obj)

            # interactive
            int_rvea = RVEA(problem=problem, interact=True, n_gen_per_iter=gen)
            int_nsga = NSGAIII(problem=problem, interact=True, n_gen_per_iter=gen)

            # initial reference point
            response = np.random.rand(n_obj)

            # run algorithms once with the randomly generated reference point
            _, pref_int_rvea = int_rvea.requests()
            _, pref_int_nsga = int_nsga.requests()
            pref_int_rvea.response = pd.DataFrame(
                [response], columns=pref_int_rvea.content["dimensions_data"].columns
            )
            pref_int_nsga.response = pd.DataFrame(
                [response], columns=pref_int_nsga.content["dimensions_data"].columns
            )

            _, pref_int_rvea = int_rvea.iterate(pref_int_rvea)
            _, pref_int_nsga = int_nsga.iterate(pref_int_nsga)

            cf = generate_composite_front(
                int_rvea.population.objectives, int_nsga.population.objectives
            )

            # ADM parameters
            L = 4
            D = 3
            lattice_resolution = 4
            reference_vectors = ReferenceVectors(lattice_resolution, n_obj)

            for i in range(L):
                data_row[["problem", "num_obj", "iteration", "num_gens"]] = [
                    problem_name,
                    n_obj,
                    i + 1,
                    gen,
                ]

                # problem_nameR = problem_name.lower()
                base = baseADM(cf, reference_vectors)

                response = gp.generateRP4learning(base)
                # print(response)
                data_row["reference_point"] = [
                    response,
                ]

                # Reference point generation for the next iteration
                pref_int_rvea.response = pd.DataFrame(
                    [response], columns=pref_int_rvea.content["dimensions_data"].columns
                )
                pref_int_nsga.response = pd.DataFrame(
                    [response], columns=pref_int_nsga.content["dimensions_data"].columns
                )

                _, pref_int_rvea = int_rvea.iterate(pref_int_rvea)
                _, pref_int_nsga = int_nsga.iterate(pref_int_nsga)

                cf = generate_composite_front(
                    cf, int_rvea.population.objectives, int_nsga.population.objectives
                )

                # R-metric calculation
                problemR = get_problem(problem_name.lower(), n_var, n_obj)
                ref_dirs = get_reference_directions(
                    "das-dennis", n_obj, n_partitions=12
                )

                ref_point = response.reshape(1, n_obj)
                rmetric = rm.RMetric(
                    problemR, ref_point, pf=problemR.pareto_front(ref_dirs)
                )

                # normalize solutions before sending r-metric

                rvea_transformer = Normalizer().fit(int_rvea.population.objectives)
                norm_rvea = rvea_transformer.transform(int_rvea.population.objectives)

                nsga_transformer = Normalizer().fit(int_nsga.population.objectives)
                norm_nsga = nsga_transformer.transform(int_nsga.population.objectives)

                cf_transformer = Normalizer().fit(cf)
                norm_cf = cf_transformer.transform(cf)

                rigd_irvea, rhv_irvea = rmetric.calc(norm_rvea, others=norm_cf)
                rigd_insga, rhv_insga = rmetric.calc(norm_nsga, others=norm_cf)

                data_row[["iRVEA" + excess_col for excess_col in excess_columns]] = [
                    rigd_irvea,
                    rhv_irvea,
                ]
                data_row[["iNSGAIII" + excess_col for excess_col in excess_columns]] = [
                    rigd_insga,
                    rhv_insga,
                ]

                data = data.append(data_row, ignore_index=1)
                fig = visualize_3D_front_rp(int_rvea.population.objectives, response)
                fig.write_html(
                    f"./results/iRVEA/" f"iRVEA_{problem_name}_iteration_{i+1}.html"
                )
                fig = visualize_3D_front_rp(int_nsga.population.objectives, response)
                fig.write_html(
                    f"./results/iNSGA/" f"iNSGA_{problem_name}_iteration_{i+1}.html"
                )

            # Decision phase
            max_assigned_vector = gp.get_max_assigned_vector(base.assigned_vectors)

            for i in range(D):
                data_row[["problem", "num_obj", "iteration", "num_gens"]] = [
                    problem_name,
                    n_obj,
                    L + i + 1,
                    gen,
                ]

                # problem_nameR = problem_name.lower()
                base = baseADM(cf, reference_vectors)

                response = gp.generateRP4decision(base, max_assigned_vector)
                # print(response)
                data_row["reference_point"] = [
                    response,
                ]

                # Reference point generation for the next iteration
                pref_int_rvea.response = pd.DataFrame(
                    [response], columns=pref_int_rvea.content["dimensions_data"].columns
                )
                pref_int_nsga.response = pd.DataFrame(
                    [response], columns=pref_int_nsga.content["dimensions_data"].columns
                )

                _, pref_int_rvea = int_rvea.iterate(pref_int_rvea)
                _, pref_int_nsga = int_nsga.iterate(pref_int_nsga)

                cf = generate_composite_front(
                    cf, int_rvea.population.objectives, int_nsga.population.objectives
                )

                # R-metric calculation
                problemR = get_problem(problem_name.lower(), n_var, n_obj)
                ref_dirs = get_reference_directions(
                    "das-dennis", n_obj, n_partitions=12
                )

                ref_point = response.reshape(1, n_obj)
                rmetric = rm.RMetric(
                    problemR, ref_point, pf=problemR.pareto_front(ref_dirs)
                )

                # normalize solutions before sending r-metric

                rvea_transformer = Normalizer().fit(int_rvea.population.objectives)
                norm_rvea = rvea_transformer.transform(int_rvea.population.objectives)

                nsga_transformer = Normalizer().fit(int_nsga.population.objectives)
                norm_nsga = nsga_transformer.transform(int_nsga.population.objectives)

                cf_transformer = Normalizer().fit(cf)
                norm_cf = cf_transformer.transform(cf)

                rigd_irvea, rhv_irvea = rmetric.calc(norm_rvea, others=norm_cf)
                rigd_insga, rhv_insga = rmetric.calc(norm_nsga, others=norm_cf)

                data_row[["iRVEA" + excess_col for excess_col in excess_columns]] = [
                    rigd_irvea,
                    rhv_irvea,
                ]
                data_row[["iNSGAIII" + excess_col for excess_col in excess_columns]] = [
                    rigd_insga,
                    rhv_insga,
                ]

                data = data.append(data_row, ignore_index=1)
                fig = visualize_3D_front_rp(int_rvea.population.objectives, response)
                fig.write_html(
                    f"./results/iRVEA/" f"iRVEA_{problem_name}_iteration_{L+i+1}.html"
                )
                fig = visualize_3D_front_rp(int_nsga.population.objectives, response)
                fig.write_html(
                    f"./results/iNSGA/" f"iNSGA_{problem_name}_iteration_{L+i+1}.html"
                )
            fig = visualize_3D_front_rvs(base.normalized_front, reference_vectors)
            fig.write_html(f"./results/" f"cf_{problem_name}.html")

data.to_csv("./results/data.csv", index=False)
