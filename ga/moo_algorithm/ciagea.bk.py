import multiprocessing
import sys
import os
import numpy as np
from typing import cast, Any
from copy import deepcopy
import random

# Add the parent directory to the module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from moo_algorithm.metric import cal_hv_front
from population import Population, Individual
from local_move import local_insert, local_flip, local_swap, local_invert
from utils import init_population


class CIAGEAPopulation(Population):
    def __init__(self, pop_size, init_div=10):
        super().__init__(pop_size)
        self.div = init_div
        self.grid_nadir = None
        self.ideal_point = None
        self.grid_indices = []
        self.ParetoFront = []
        self.archive = []
        self.archive_size = min(50, pop_size // 2)

    def update_archive(self, new_solutions):
        """Maintain archive of best solutions"""
        # Combine archive with new solutions
        combined = self.archive + new_solutions

        # Fast non-dominated sort
        fronts = self.fast_nondominated_sort(combined)

        # Keep only first front, limited by archive size
        self.archive = fronts[0][: self.archive_size] if fronts else []

    def check_diversity(self):
        """Check if population has converged"""
        if len(self.indivs) < 2:
            return True

        # Check grid diversity
        unique_grids = len(set(self.grid_indices))
        diversity_ratio = unique_grids / len(self.grid_indices)

        # FIXED: Lower threshold from 0.3 to 0.2
        return diversity_ratio > 0.2

    def update_ideal_point(self, solutions):
        """Update ideal point from solutions"""
        if not solutions:
            return

        M = len(solutions[0].objectives)
        if self.ideal_point is None:
            self.ideal_point = np.array([np.inf] * M)

        for sol in solutions:
            for j in range(M):
                self.ideal_point[j] = min(self.ideal_point[j], sol.objectives[j])

    def update_grid_nadir(self, nd_solutions):
        """Update grid nadir point with stabilization strategy"""
        if not nd_solutions:
            return

        M = len(nd_solutions[0].objectives)

        # Calculate current nadir point from non-dominated solutions
        z_nadir = np.array([-np.inf] * M)
        for sol in nd_solutions:
            for j in range(M):
                z_nadir[j] = max(z_nadir[j], sol.objectives[j])

        # Initialize grid nadir if first time
        if self.grid_nadir is None:
            self.grid_nadir = z_nadir.copy()
            return

        # Calculate grid spacing
        grid_spacing = (self.grid_nadir - self.ideal_point) / (self.div - 1)

        # Update grid nadir with stabilization strategy (Eq. 6)
        for j in range(M):
            if abs(z_nadir[j] - self.grid_nadir[j]) > grid_spacing[j] / 2:
                self.grid_nadir[j] = z_nadir[j]

    def calculate_grid_spacing(self):
        """Calculate grid spacing (Eq. 4)"""
        if self.grid_nadir is None or self.ideal_point is None:
            return (
                np.zeros(len(self.indivs[0].objectives)) if self.indivs else np.zeros(2)
            )

        spacing = (self.grid_nadir - self.ideal_point) / (self.div - 1)
        # Ensure minimum spacing to avoid division by zero
        spacing = np.maximum(spacing, 1e-10)
        return spacing

    def calculate_lower_boundaries(self, grid_spacing):
        """Calculate lower boundaries (Eq. 5)"""
        return self.ideal_point - grid_spacing / 2

    def calculate_grid_index(self, solution, lower_boundaries, grid_spacing):
        """Calculate grid index for a solution (Eq. 7)"""
        M = len(solution.objectives)
        grid_index = []
        for j in range(M):
            # Handle zero or very small grid spacing
            if grid_spacing[j] < 1e-10:
                idx = 0
            else:
                idx = int(
                    (solution.objectives[j] - lower_boundaries[j]) / grid_spacing[j]
                )
            grid_index.append(max(0, idx))  # Ensure non-negative
        return tuple(grid_index)

    def calculate_grid_corner(self, grid_index, lower_boundaries, grid_spacing):
        """Calculate grid corner position (Eq. 8)"""
        M = len(grid_index)
        corner = []
        for j in range(M):
            corner.append(lower_boundaries[j] + grid_index[j] * grid_spacing[j])
        return np.array(corner)

    def normalize_objectives(self, objectives):
        """Normalize objectives (Eq. 9)"""
        if self.grid_nadir is None or self.ideal_point is None:
            return objectives
        return (objectives - self.ideal_point) / (
            self.grid_nadir - self.ideal_point + 1e-10
        )

    def calculate_fitness(self, solution, grid_corner, grid_index):
        """Calculate fitness for environmental selection (Eq. 11)"""
        normalized_obj = self.normalize_objectives(np.array(solution.objectives))
        normalized_corner = self.normalize_objectives(grid_corner)

        M = len(solution.objectives)
        distance_sum = 0

        for j in range(M):
            diff = normalized_obj[j] - normalized_corner[j]
            # Apply boundary protection: amplify difference if on lower boundary
            if grid_index[j] == 0:
                diff *= 1e6
            distance_sum += diff**2

        return np.sqrt(distance_sum)

    def environmental_selection(self, solutions):
        """Environmental selection with boundary solution protection"""
        if not solutions:
            return [], []

        grid_spacing = self.calculate_grid_spacing()
        lower_boundaries = self.calculate_lower_boundaries(grid_spacing)

        # Calculate grid index and fitness for each solution
        solution_data = []
        for sol in solutions:
            grid_idx = self.calculate_grid_index(sol, lower_boundaries, grid_spacing)
            grid_corner = self.calculate_grid_corner(
                grid_idx, lower_boundaries, grid_spacing
            )
            fitness = self.calculate_fitness(sol, grid_corner, grid_idx)
            solution_data.append((sol, grid_idx, fitness))

        # Group solutions by grid index
        grid_dict = {}
        for sol, grid_idx, fitness in solution_data:
            if grid_idx not in grid_dict:
                grid_dict[grid_idx] = []
            grid_dict[grid_idx].append((sol, fitness))

        # Select one solution per subspace (lowest fitness)
        selected_solutions = []
        selected_indices = []
        for grid_idx, sols_fitness in grid_dict.items():
            # Sort by fitness and select the best one
            sols_fitness.sort(key=lambda x: x[1])
            selected_solutions.append(sols_fitness[0][0])
            selected_indices.append(grid_idx)

        return selected_solutions, selected_indices

    def adaptive_grid_adjustment(self, solutions):
        """Adaptive grid divisions adjustment strategy"""
        # Perform environmental selection
        selected, indices = self.environmental_selection(solutions)

        if len(selected) > self.pop_size:
            # Too many solutions, decrease divisions
            self.div = max(2, self.div - 1)
            selected_new, indices_new = self.environmental_selection(solutions)

            if len(selected_new) >= self.pop_size:
                return selected_new, indices_new
            else:
                # Revert
                self.div += 1
                return selected, indices

        elif len(selected) < self.pop_size:
            # Too few solutions, increase divisions
            self.div += 1

        return selected, indices

    def calculate_crowding_degree(self, grid_indices):
        """Calculate crowding degree for each solution (Eq. 12, 13)"""
        crowding = []
        M = len(grid_indices[0]) if grid_indices else 0

        for i, idx_i in enumerate(grid_indices):
            neighbor_count = 0
            for j, idx_j in enumerate(grid_indices):
                if i == j:
                    continue
                # Check if neighbor (Chebyshev distance = 1)
                max_diff = max(abs(idx_i[k] - idx_j[k]) for k in range(M))
                if max_diff == 1:
                    neighbor_count += 1
            crowding.append(neighbor_count)

        return crowding

    def population_reselection(self, solutions, grid_indices):
        """Population reselection based on crowding degree"""
        if len(solutions) <= self.pop_size:
            return solutions

        M = len(grid_indices[0]) if grid_indices else 1
        crowding = self.calculate_crowding_degree(grid_indices)

        # Calculate fitness (Eq. 14)
        max_crowding = max(crowding) if crowding else 1

        # Handle case when all solutions have 0 neighbors
        if max_crowding == 0:
            # Random selection when all equally sparse
            selected_idx = np.random.choice(
                len(solutions), size=self.pop_size, replace=False
            )
            return [solutions[int(i)] for i in cast(Any, selected_idx)]

        fitness = []
        for c in crowding:
            f = (self.pop_size - 1) * (c ** (1 / M)) / (max_crowding ** (1 / M)) + 1
            fitness.append(f)

        # Roulette wheel selection
        fitness = np.array(fitness)
        # Invert fitness (lower crowding = higher selection probability)
        fitness = max(fitness) - fitness + 1
        probs = fitness / fitness.sum()

        selected_idx = np.random.choice(
            len(solutions), size=self.pop_size, replace=False, p=probs
        )
        return [solutions[i] for i in cast(Any, selected_idx)]

    def fast_nondominated_sort(self, solutions):
        """Fast non-dominated sorting"""
        ParetoFront = [[]]
        for individual in solutions:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in solutions:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                ParetoFront[0].append(individual)

        i = 0
        while len(ParetoFront[i]) > 0:
            temp = []
            for individual in ParetoFront[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i += 1
            ParetoFront.append(temp)

        return ParetoFront

    def adaptive_local_search(
        self, individual, problem, generation, max_gen, current_population
    ):
        """Apply local search with FIXED issues from original"""

        # FIXED: Slower decay - was too aggressive
        ls_probability = 0.6 * (1 - (generation / max_gen) ** 0.35)

        if np.random.random() > ls_probability:
            return individual

        # Grid-based selection - only apply LS to sparse regions
        if hasattr(individual, "objectives") and individual.objectives:
            grid_spacing = self.calculate_grid_spacing()
            lower_boundaries = self.calculate_lower_boundaries(grid_spacing)
            ind_grid = self.calculate_grid_index(
                individual, lower_boundaries, grid_spacing
            )

            grid_count = sum(1 for idx in self.grid_indices if idx == ind_grid)

            # FIXED: More lenient threshold - was skipping too many
            if grid_count > 4:  # Changed from 3
                return individual

        # Multi-start local search
        best_improved = Individual(deepcopy(individual.chromosome))

        # FIXED: More exploration attempts
        num_tries = 2  # Increased from 1-2 variable

        for _ in range(num_tries):
            current = Individual(deepcopy(individual.chromosome))

            # FIXED: Better depth scaling
            if generation < max_gen * 0.4:  # Extended early phase
                depth = random.randint(3, 5)
            else:
                depth = random.randint(2, 4)  # Increased minimum

            for step in range(depth):
                # Mix of different operators for diversity
                move_probs = [0.3, 0.3, 0.2, 0.2]  # swap, insert, invert, flip
                move_type = np.random.choice(
                    ["swap", "insert", "invert", "flip"], p=move_probs
                )

                if move_type == "swap":
                    current = local_swap(current)
                elif move_type == "insert":
                    current = local_insert(current)
                elif move_type == "invert":
                    current = local_invert(current)
                else:  # flip
                    current = local_flip(current, problem)

            best_improved = current

        return best_improved


def run_ciagea(
    processing_number,
    problem,
    indi_list,
    pop_size,
    max_gen,
    init_div,
    crossover_operator,
    mutation_operator,
    crossover_rate,
    mutation_rate,
    cal_fitness,
):
    print("CIAGEA (Minimal Fixed)")
    history = {}

    ciagea_pop = CIAGEAPopulation(pop_size, init_div)
    ciagea_pop.pre_indi_gen(indi_list)

    pool = multiprocessing.Pool(processing_number)

    # Initial evaluation
    arg = [(problem, individual) for individual in ciagea_pop.indivs]
    result = pool.starmap(cal_fitness, arg)
    for individual, fitness in zip(ciagea_pop.indivs, result):
        individual.chromosome = fitness[0]
        individual.objectives = fitness[1:]

    # Update ideal point
    ciagea_pop.update_ideal_point(ciagea_pop.indivs)

    # Non-dominated sorting to get non-dominated solutions
    fronts = ciagea_pop.fast_nondominated_sort(ciagea_pop.indivs)
    nd_solutions = fronts[0] if fronts else []

    # Update grid nadir
    ciagea_pop.update_grid_nadir(nd_solutions)

    # Adaptive grid adjustment and environmental selection
    selected, grid_indices = ciagea_pop.adaptive_grid_adjustment(ciagea_pop.indivs)
    ciagea_pop.indivs = selected
    ciagea_pop.grid_indices = grid_indices

    # Store initial Pareto front
    ciagea_pop.ParetoFront = [fronts[0]] if fronts else [[]]
    Pareto_store = [list(indi.objectives) for indi in ciagea_pop.ParetoFront[0]]
    history[0] = [Pareto_store, ciagea_pop.div]
    print("Generation 0: Done")

    # Evolution loop
    for gen in range(max_gen):
        print(f"generation {gen}: {len(ciagea_pop.indivs)}")
        # ========================================================================
        # DIVERSITY RESTART - FIXED: Less aggressive
        # ========================================================================
        # FIXED: Changed from every 20 to every 35 generations
        if gen % 35 == 0 and gen > 0:
            if not ciagea_pop.check_diversity():
                print(f"  Diversity restart at generation {gen}")

                # FIXED: Reduced from 1/3 to 1/5 of population
                num_new = pop_size // 5
                new_indivs = init_population(num_new, 42 + gen, problem)

                # Evaluate new individuals
                arg = [(problem, individual) for individual in new_indivs]
                result = pool.starmap(cal_fitness, arg)
                for individual, fitness in zip(new_indivs, result):
                    individual.chromosome = fitness[0]
                    individual.objectives = fitness[1:]

                # Combine with current population
                combined = ciagea_pop.indivs + new_indivs

                # Update ideal point with combined population
                ciagea_pop.update_ideal_point(combined)

                # Re-run environmental selection
                selected, grid_indices = ciagea_pop.adaptive_grid_adjustment(combined)

                # Ensure we keep pop_size individuals
                if len(selected) > pop_size:
                    ciagea_pop.indivs = ciagea_pop.population_reselection(
                        selected, grid_indices
                    )
                else:
                    ciagea_pop.indivs = selected

                ciagea_pop.grid_indices = grid_indices[: len(ciagea_pop.indivs)]

                print(
                    f"  Diversity restored: {len(set(ciagea_pop.grid_indices))} unique grid locations"
                )

        # ========================================================================
        # NORMAL EVOLUTION
        # ========================================================================

        # Generate offspring
        offspring = ciagea_pop.gen_offspring(
            problem,
            crossover_operator,
            mutation_operator,
            crossover_rate,
            mutation_rate,
        )

        # Apply local search to some offspring
        # FIXED: Reduced from 40% to 30%
        ls_offspring = []
        for off in offspring:
            if np.random.random() < 0.3:
                improved = ciagea_pop.adaptive_local_search(
                    off, problem, gen, max_gen, None
                )
                ls_offspring.append(improved)
            else:
                ls_offspring.append(off)

        # Evaluate offspring
        arg = [(problem, individual) for individual in ls_offspring]
        result = pool.starmap(cal_fitness, arg)
        for individual, fitness in zip(ls_offspring, result):
            individual.chromosome = fitness[0]
            individual.objectives = fitness[1:]

        # FIXED: Actually use archive
        ciagea_pop.update_archive(ls_offspring)

        # Combine population and offspring
        combined = ciagea_pop.indivs + ls_offspring

        # Update ideal point
        ciagea_pop.update_ideal_point(combined)

        # Non-dominated sorting
        fronts = ciagea_pop.fast_nondominated_sort(combined)

        # Select solutions that can enter grid (first N_pop from fronts)
        grid_candidates = []
        for front in fronts:
            if len(grid_candidates) + len(front) <= int(pop_size * 1.5):
                grid_candidates.extend(front)
            else:
                remaining = int(pop_size * 1.5) - len(grid_candidates)
                grid_candidates.extend(front[:remaining])
                break

        nd_solutions = fronts[0] if fronts else []

        # Update grid nadir
        ciagea_pop.update_grid_nadir(nd_solutions)

        # Adaptive grid adjustment and environmental selection
        selected, grid_indices = ciagea_pop.adaptive_grid_adjustment(grid_candidates)

        # Population reselection
        ciagea_pop.indivs = ciagea_pop.population_reselection(selected, grid_indices)
        ciagea_pop.grid_indices = grid_indices[: len(ciagea_pop.indivs)]

        # Update Pareto front
        ciagea_pop.ParetoFront = [fronts[0]] if fronts else [[]]

        print(f"Generation {gen + 1}: Done")
        Pareto_store = [list(indi.objectives) for indi in ciagea_pop.ParetoFront[0]]
        history[gen + 1] = [Pareto_store, ciagea_pop.div]

    pool.close()
    print(
        "CIAGEA Done: ", cal_hv_front(ciagea_pop.ParetoFront[0], np.array([10, 100000]))
    )
    return history
