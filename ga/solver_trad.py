import multiprocessing
import random
import numpy as np
import os
import sys
import json
from copy import deepcopy
from typing import List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem import Problem, Node
from population import Individual, Population
from utils import (
    partition,
    unset,
    dronable,
    balance,
    overload,
    routing,
    schedule,
    cal_route_distance,
    cal_tardiness,
    cal_cost,
    Solution,
)
from operators import crossover_PMX, mutation_flip

# --------------------------------------------------------------------------- #
#  Scalarisation weights                                                        #
# --------------------------------------------------------------------------- #
W_TARDINESS = 0.5
W_COST = 0.5


def scalarize(tardiness: float, cost: float) -> float:
    return W_TARDINESS * tardiness + W_COST * cost


# --------------------------------------------------------------------------- #
#  Population initialisation                                                    #
# --------------------------------------------------------------------------- #

def _split_nodes(problem: Problem):
    """
    Trả về (lh_ids, bh_ids) – danh sách id node linehaul và backhaul.
    Node 0 là depot, bỏ qua.
    """
    lh_ids, bh_ids = [], []
    for node in problem.nodes[1:]:          # bỏ depot
        if node.demand >= 0:                # LINEHAUL: demand > 0 (giao hàng) hoặc 0
            lh_ids.append(node.id)
        else:                               # BACKHAUL: demand < 0 (thu hàng)
            bh_ids.append(node.id)
    return lh_ids, bh_ids


def _make_half_chro(node_ids: List[int], num_fleet: int) -> List:
    """
    Tạo một nửa chromosome (perm + mask) cho tập node_ids.
    Delimiter id = n_node, n_node+1, … (như utils.py).
    """
    n_node_real = len(node_ids)
    if n_node_real == 0:
        return [[], []]

    # delimiter ids bắt đầu từ max(node_ids)+1 để tránh trùng
    # Thực ra utils dùng range(1, n_nodes+n_fleet) nên ta làm tương tự:
    # delimiter >= n_node (len(problem.nodes))
    # Ở đây ta không biết n_node từ problem nên caller truyền vào delimiter_start
    raise NotImplementedError("Dùng init_half_chro thay thế")


def init_half_chro(node_ids: List[int], num_fleet: int, delimiter_start: int) -> List:
    """
    Tạo một nửa chromosome ngẫu nhiên cho tập node_ids.

    perm  = hoán vị ngẫu nhiên của node_ids + (num_fleet-1) delimiters
    mask  = vector {0,+1,-1} ngẫu nhiên cùng độ dài perm
    """
    if not node_ids:
        return [[], []]

    delimiters = list(range(delimiter_start, delimiter_start + num_fleet - 1))
    pool = node_ids + delimiters
    perm = random.sample(pool, len(pool))
    mask = random.choices([0, 1, -1], weights=[0.7, 0.15, 0.15], k=len(pool))
    # delimiter không có mask ý nghĩa → đặt 0
    for i, g in enumerate(perm):
        if g >= delimiter_start:
            mask[i] = 0
    return [perm, mask]


def init_population_trad(num_indi: int, seed: int, problem: Problem) -> List[Individual]:
    """
    Khởi tạo quần thể cho traditional-backhaul GA.

    Mỗi Individual.chromosome = [lh_chro, bh_chro]
        lh_chro = [lh_perm, lh_mask]
        bh_chro = [bh_perm, bh_mask]
    """
    random.seed(seed)
    lh_ids, bh_ids = _split_nodes(problem)
    n_node = len(problem.nodes)          # depot + tất cả node → delimiter >= n_node
    num_fleet = problem.num_fleet

    population = []
    for _ in range(num_indi):
        lh_chro = init_half_chro(lh_ids, num_fleet, delimiter_start=n_node)
        bh_chro = init_half_chro(bh_ids, num_fleet, delimiter_start=n_node)
        indi = Individual([lh_chro, bh_chro])
        population.append(indi)
    return population


# --------------------------------------------------------------------------- #
#  Repair                                                                       #
# --------------------------------------------------------------------------- #

def repair_half(chro: List, problem: Problem):
    """
    Áp dụng repair cho một nửa chromosome (giống repair() trong utils.py).
    unset → balance nếu overload → dronable
    """
    unset(chro, problem)
    if overload(chro, problem):
        chro[:] = balance(chro, problem)
    dronable(chro, problem)


def repair_trad(chromosome: List, problem: Problem):
    """Repair toàn bộ chromosome [lh_chro, bh_chro]."""
    lh_chro, bh_chro = chromosome
    repair_half(lh_chro, problem)
    repair_half(bh_chro, problem)


# --------------------------------------------------------------------------- #
#  Decode                                                                       #
# --------------------------------------------------------------------------- #

def _align_seqs(lh_seqs: List, bh_seqs: List, num_fleet: int) -> List[Tuple]:
    """
    Căn chỉnh hai danh sách sequence về đúng num_fleet phần tử.
    Thiếu thì thêm list rỗng, thừa thì gộp vào cuối.
    Trả về list của (lh_seq, bh_seq) theo từng route.
    """
    # Pad / trim
    while len(lh_seqs) < num_fleet:
        lh_seqs.append([])
    while len(bh_seqs) < num_fleet:
        bh_seqs.append([])

    # Nếu thừa → gộp vào route cuối
    if len(lh_seqs) > num_fleet:
        extra = []
        for s in lh_seqs[num_fleet:]:
            extra.extend(s)
        lh_seqs[num_fleet - 1].extend(extra)
        lh_seqs = lh_seqs[:num_fleet]

    if len(bh_seqs) > num_fleet:
        extra = []
        for s in bh_seqs[num_fleet:]:
            extra.extend(s)
        bh_seqs[num_fleet - 1].extend(extra)
        bh_seqs = bh_seqs[:num_fleet]

    return list(zip(lh_seqs[:num_fleet], bh_seqs[:num_fleet]))


def decode_trad(indi: Individual, problem: Problem) -> Tuple[Solution, Solution]:
    """
    Decode chromosome [lh_chro, bh_chro] thành Solution theo traditional backhaul.

    Với mỗi route k:  seq = lh_seqs[k]  +  bh_seqs[k]   (linehaul trước)
    Sau đó gọi routing() → schedule() y hệt utils.decode().
    """
    lh_chro, bh_chro = indi.chromosome

    lh_seqs = partition(lh_chro, problem) if lh_chro[0] else []
    bh_seqs = partition(bh_chro, problem) if bh_chro[0] else []

    paired = _align_seqs(lh_seqs, bh_seqs, problem.num_fleet)

    tardiness_routes = []
    cost_routes = []
    failed_indices = []

    for k, (lh_seq, bh_seq) in enumerate(paired):
        # Ghép linehaul trước, backhaul sau
        seq = lh_seq + bh_seq

        if not seq:
            # Route rỗng → tạo route depot→depot rỗng
            from utils import Route
            empty_route = Route(nodes=[0, len(problem.nodes)],
                                arrival=[0.0, 0.0],
                                departure=[0.0, 0.0],
                                service=[0.0, 0.0])
            tardiness_routes.append((empty_route, []))
            cost_routes.append((empty_route, []))
            continue

        t_route, t_trips, c_route, c_trips = routing(seq, problem)

        if not t_route:
            # Nếu routing thất bại (không thể lập lịch) → đặt mask về 0
            for i in range(len(seq)):
                seq[i] = (seq[i][0], 0)
            failed_indices.append(k)
            t_route, t_trips, c_route, c_trips = routing(seq, problem)

        tardiness_routes.append((t_route, t_trips))
        cost_routes.append((c_route, c_trips))

    # Cập nhật lại mask trong chromosome nếu có route thất bại
    if failed_indices:
        _clear_failed_masks(indi.chromosome, problem, failed_indices)

    return Solution(tardiness_routes), Solution(cost_routes)


def _clear_failed_masks(chromosome: List, problem: Problem, failed_indices: List[int]):
    """Đặt lại mask về 0 cho các route thất bại trong cả hai nửa chromosome."""
    n_node = len(problem.nodes)

    for chro in chromosome:   # lh_chro, bh_chro
        if not chro[0]:
            continue
        seq_idx = 0
        start = 0
        for i in range(len(chro[0])):
            if chro[0][i] >= n_node:
                if i > start:
                    if seq_idx in failed_indices:
                        for j in range(start, i):
                            chro[1][j] = 0
                    seq_idx += 1
                start = i + 1
            elif i == len(chro[0]) - 1:
                if start < len(chro[0]):
                    if seq_idx in failed_indices:
                        for j in range(start, len(chro[0])):
                            chro[1][j] = 0
                    seq_idx += 1


# --------------------------------------------------------------------------- #
#  Fitness                                                                      #
# --------------------------------------------------------------------------- #

def cal_fitness_trad(problem: Problem, indi: Individual):
    """
    Tính fitness cho traditional-backhaul individual.
    Trả về (chromosome, tardiness, cost) – khớp với cal_fitness() trong utils.
    """
    repair_trad(indi.chromosome, problem)
    tardiness_sol, cost_sol = decode_trad(indi, problem)
    tardiness = cal_tardiness(tardiness_sol, problem)
    cost = cal_cost(cost_sol, problem)
    return indi.chromosome, tardiness, cost


def cal_scalar_fitness_trad(problem: Problem, indi: Individual):
    chro, tardiness, cost = cal_fitness_trad(problem, indi)
    scalar = scalarize(tardiness, cost)
    return chro, scalar, tardiness, cost


# --------------------------------------------------------------------------- #
#  Crossover & Mutation cho traditional-backhaul                               #
# --------------------------------------------------------------------------- #

def crossover_trad_PMX(problem: Problem,
                       parent1: Individual,
                       parent2: Individual) -> Tuple[Individual, Individual]:
    """
    PMX trên từng nửa chromosome (lh và bh) độc lập.
    Tái sử dụng crossover_PMX() từ operators.py bằng cách đóng gói tạm.
    """
    # Tạo Individual giả chỉ chứa một nửa để dùng crossover_PMX
    def _half_individual(chro_half):
        ind = Individual(chro_half)
        return ind

    lh1 = _half_individual(deepcopy(parent1.chromosome[0]))
    lh2 = _half_individual(deepcopy(parent2.chromosome[0]))
    bh1 = _half_individual(deepcopy(parent1.chromosome[1]))
    bh2 = _half_individual(deepcopy(parent2.chromosome[1]))

    # Crossover nửa linehaul
    if lh1.chromosome[0]:   # không rỗng
        off_lh1, off_lh2 = crossover_PMX(problem, lh1, lh2)
        new_lh1, new_lh2 = off_lh1.chromosome, off_lh2.chromosome
    else:
        new_lh1, new_lh2 = deepcopy(lh1.chromosome), deepcopy(lh2.chromosome)

    # Crossover nửa backhaul
    if bh1.chromosome[0]:   # không rỗng
        off_bh1, off_bh2 = crossover_PMX(problem, bh1, bh2)
        new_bh1, new_bh2 = off_bh1.chromosome, off_bh2.chromosome
    else:
        new_bh1, new_bh2 = deepcopy(bh1.chromosome), deepcopy(bh2.chromosome)

    child1 = Individual([new_lh1, new_bh1])
    child2 = Individual([new_lh2, new_bh2])
    return child1, child2


def mutation_trad_flip(problem: Problem, indi: Individual) -> Individual:
    """
    Flip mutation trên từng nửa chromosome độc lập.
    """
    def _half_individual(chro_half):
        return Individual(chro_half)

    new_lh = deepcopy(indi.chromosome[0])
    new_bh = deepcopy(indi.chromosome[1])

    if new_lh[0]:
        tmp = mutation_flip(problem, _half_individual(new_lh))
        new_lh = tmp.chromosome

    if new_bh[0]:
        tmp = mutation_flip(problem, _half_individual(new_bh))
        new_bh = tmp.chromosome

    return Individual([new_lh, new_bh])


# --------------------------------------------------------------------------- #
#  GA Population                                                                #
# --------------------------------------------------------------------------- #

class TradGAPopulation(Population):
    def __init__(self, pop_size: int):
        super().__init__(pop_size)

    def tournament_selection(self, k: int = 3) -> Individual:
        """k-tournament selection (lower scalar fitness wins)."""
        candidates = random.sample(self.indivs, min(k, len(self.indivs)))
        return min(candidates, key=lambda ind: ind.scalar_fitness)

    def gen_offspring_ga(self,
                         problem: Problem,
                         crossover_operator,
                         mutation_operator,
                         crossover_rate: float,
                         mutation_rate: float) -> List[Individual]:
        offspring = []
        while len(offspring) < self.pop_size:
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            if np.random.rand() < crossover_rate:
                off1, off2 = crossover_operator(problem, p1, p2)
            else:
                off1 = Individual(deepcopy(p1.chromosome))
                off2 = Individual(deepcopy(p2.chromosome))
            if np.random.rand() < mutation_rate:
                off1 = mutation_operator(problem, off1)
            if np.random.rand() < mutation_rate:
                off2 = mutation_operator(problem, off2)
            offspring.append(off1)
            offspring.append(off2)
        return offspring[:self.pop_size]

    def natural_selection(self, combined: List[Individual]):
        """(μ+λ) survivor selection – giữ pop_size cá thể tốt nhất."""
        combined.sort(key=lambda ind: ind.scalar_fitness)
        self.indivs = combined[:self.pop_size]


# --------------------------------------------------------------------------- #
#  Evaluation helper                                                            #
# --------------------------------------------------------------------------- #

def _evaluate_population(pool, problem: Problem, individuals: List[Individual]):
    args = [(problem, ind) for ind in individuals]
    results = pool.starmap(cal_scalar_fitness_trad, args)
    for ind, res in zip(individuals, results):
        chro, scalar, tardiness, cost = res
        ind.chromosome = chro
        ind.scalar_fitness = scalar
        ind.objectives = [tardiness, cost]
    return individuals


# --------------------------------------------------------------------------- #
#  Main GA runner                                                               #
# --------------------------------------------------------------------------- #

def run_ga_trad(processing_number: int,
                problem: Problem,
                indi_list: List[Individual],
                pop_size: int,
                max_gen: int,
                crossover_operator=crossover_trad_PMX,
                mutation_operator=mutation_trad_flip,
                crossover_rate: float = 0.9,
                mutation_rate: float = 0.1,
                verbose: bool = True) -> dict:
    print("GA Traditional Backhaul (w_tardiness=0.5, w_cost=0.5)")

    ga_pop = TradGAPopulation(pop_size)
    ga_pop.pre_indi_gen(indi_list)

    history = {}
    pool = multiprocessing.Pool(processing_number)

    # --- Generation 0 ---
    _evaluate_population(pool, problem, ga_pop.indivs)
    ga_pop.indivs.sort(key=lambda ind: ind.scalar_fitness)

    best = deepcopy(ga_pop.indivs[0])
    history[0] = _snapshot(ga_pop.indivs[0])

    if verbose:
        _print_gen(0, ga_pop.indivs[0])

    # --- Evolution loop ---
    for gen in range(1, max_gen + 1):
        offspring = ga_pop.gen_offspring_ga(
            problem,
            crossover_operator,
            mutation_operator,
            crossover_rate,
            mutation_rate,
        )

        _evaluate_population(pool, problem, offspring)

        combined = ga_pop.indivs + offspring
        ga_pop.natural_selection(combined)

        if ga_pop.indivs[0].scalar_fitness < best.scalar_fitness:
            best = deepcopy(ga_pop.indivs[0])

        history[gen] = _snapshot(ga_pop.indivs[0])

        if verbose:
            _print_gen(gen, ga_pop.indivs[0])

    pool.close()
    pool.join()

    print(
        f"\nGA Trad Done | best scalar={best.scalar_fitness:.4f} "
        f"| tardiness={best.objectives[0]:.4f} | cost={best.objectives[1]:.2f}"
    )
    return {"history": history, "best": best}


# --------------------------------------------------------------------------- #
#  Utilities                                                                    #
# --------------------------------------------------------------------------- #

def _snapshot(ind: Individual) -> dict:
    return {
        "scalar": ind.scalar_fitness,
        "tardiness": ind.objectives[0],
        "cost": ind.objectives[1],
    }


def _print_gen(gen: int, best_ind: Individual):
    print(
        f"  Gen {gen:>4d} | scalar={best_ind.scalar_fitness:>12.4f} "
        f"| tardiness={best_ind.objectives[0]:>8.4f} "
        f"| cost={best_ind.objectives[1]:>10.2f}"
    )


# --------------------------------------------------------------------------- #
#  Entry point (example usage)                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "../data/generated/data/N10/S042_N10_C_R50.json"
    POP_SIZE = 50
    MAX_GEN = 100
    N_WORKERS = max(1, multiprocessing.cpu_count() - 1)
    SEED = 42

    problem = Problem(DATA_PATH)
    indi_list = init_population_trad(POP_SIZE, SEED, problem)

    result = run_ga_trad(
        processing_number=N_WORKERS,
        problem=problem,
        indi_list=indi_list,
        pop_size=POP_SIZE,
        max_gen=MAX_GEN,
        crossover_operator=crossover_trad_PMX,
        mutation_operator=mutation_trad_flip,
        crossover_rate=0.9,
        mutation_rate=0.1,
        verbose=True,
    )

    base_filename = os.path.splitext(os.path.basename(DATA_PATH))[0]
    parent_dir = os.path.basename(os.path.dirname(os.path.normpath(DATA_PATH)))
    output_dir = os.path.join("result_ga_trad", parent_dir)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{base_filename}.json")

    with open(out_path, "w") as f:
        json.dump(result["history"], f, indent=2)

    print(f"\n[OK] History saved to: {out_path}")