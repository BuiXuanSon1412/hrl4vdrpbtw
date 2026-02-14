from moo_algorithm.moead import run_moead, init_weight_vectors_2d
from moo_algorithm.nsga_ii import run_nsga_ii
from moo_algorithm.nsga_iii import run_nsga_iii
from moo_algorithm.pfg_moea import run_pfgmoea
from moo_algorithm.agea import run_agea
from moo_algorithm.ciagea import run_ciagea
from operators import crossover_PMX, mutation_flip

ALGORITHMS = {
    "CIAGEA": {
        "runner": run_ciagea,
        "params": {
            "init_div": 10,
            "crossover_operator": crossover_PMX,
            "mutation_operator": mutation_flip,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1,
        },
        "ref_point": [24, 100000],
    },
    "AGEA": {
        "runner": run_agea,
        "params": {
            "init_div": 10,
            "crossover_operator": crossover_PMX,
            "mutation_operator": mutation_flip,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1,
        },
        "ref_point": [24, 100000],
    },
    "PFG_MOEA": {
        "runner": run_pfgmoea,
        "params": {
            "GK": 10,
            "sigma": 0.1,
            "crossover_operator": crossover_PMX,
            "mutation_operator": mutation_flip,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1,
        },
        "ref_point": [24, 100000],
    },
    "MOEAD": {
        "runner": run_moead,
        "params": {
            "neighborhood_size": 20,
            "init_weight_vectors": init_weight_vectors_2d,
            "crossover_operator": crossover_PMX,
            "mutation_operator": mutation_flip,
        },
        "ref_point": [24, 100000],
    },
    "NSGA_II": {
        "runner": run_nsga_ii,
        "params": {
            "crossover_operator": crossover_PMX,
            "mutation_operator": mutation_flip,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1,
        },
        "ref_point": [24, 100000],
    },
    "NSGA_III": {
        "runner": run_nsga_iii,
        "params": {
            "crossover_operator": crossover_PMX,
            "mutation_operator": mutation_flip,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1,
        },
        "ref_point": [24, 100000],
    },
}
