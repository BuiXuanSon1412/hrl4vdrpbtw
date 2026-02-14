import multiprocessing
import sys
import os
import numpy as np
from typing import cast, Any

# Add the parent directory to the module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from moo_algorithm.metric import cal_hv_front
from population import Population


class AGEAPopulation(Population):
    def __init__(self, pop_size, init_div=10):
        super().__init__(pop_size)
        self.div = init_div
        self.grid_nadir = None
        self.ideal_point = None
        self.grid_indices = []
        self.ParetoFront = []

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


def run_agea(
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
    print("AGEA")
    history = {}

    agea_pop = AGEAPopulation(pop_size, init_div)
    agea_pop.pre_indi_gen(indi_list)

    pool = multiprocessing.Pool(processing_number)

    # Initial evaluation
    arg = [(problem, individual) for individual in agea_pop.indivs]
    result = pool.starmap(cal_fitness, arg)
    for individual, fitness in zip(agea_pop.indivs, result):
        individual.chromosome = fitness[0]
        individual.objectives = fitness[1:]

    # Update ideal point
    agea_pop.update_ideal_point(agea_pop.indivs)

    # Non-dominated sorting to get non-dominated solutions
    fronts = agea_pop.fast_nondominated_sort(agea_pop.indivs)
    nd_solutions = fronts[0] if fronts else []

    # Update grid nadir
    agea_pop.update_grid_nadir(nd_solutions)

    # Adaptive grid adjustment and environmental selection
    selected, grid_indices = agea_pop.adaptive_grid_adjustment(agea_pop.indivs)
    agea_pop.indivs = selected
    agea_pop.grid_indices = grid_indices

    # Store initial Pareto front
    agea_pop.ParetoFront = [fronts[0]] if fronts else [[]]
    Pareto_store = [list(indi.objectives) for indi in agea_pop.ParetoFront[0]]
    history[0] = Pareto_store
    print("Generation 0: Done")

    # Evolution loop
    for gen in range(max_gen):
        print(
            f"generation {gen}: (pop_size: {len(agea_pop.indivs)}), (div: {agea_pop.div})"
        )
        # Generate offspring
        offspring = agea_pop.gen_offspring(
            problem,
            crossover_operator,
            mutation_operator,
            crossover_rate,
            mutation_rate,
        )

        # Evaluate offspring
        arg = [(problem, individual) for individual in offspring]
        result = pool.starmap(cal_fitness, arg)
        for individual, fitness in zip(offspring, result):
            individual.chromosome = fitness[0]
            individual.objectives = fitness[1:]

        # Combine population and offspring
        combined = agea_pop.indivs + offspring

        # Update ideal point
        agea_pop.update_ideal_point(combined)

        # Non-dominated sorting
        fronts = agea_pop.fast_nondominated_sort(combined)

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
        agea_pop.update_grid_nadir(nd_solutions)

        # Adaptive grid adjustment and environmental selection
        selected, grid_indices = agea_pop.adaptive_grid_adjustment(grid_candidates)

        # Population reselection
        agea_pop.indivs = agea_pop.population_reselection(selected, grid_indices)
        agea_pop.grid_indices = grid_indices[: len(agea_pop.indivs)]

        # Update Pareto front
        agea_pop.ParetoFront = [fronts[0]] if fronts else [[]]

        print(f"Generation {gen + 1}: Done")
        Pareto_store = [list(indi.objectives) for indi in agea_pop.ParetoFront[0]]
        history[gen + 1] = Pareto_store

    pool.close()
    print("AGEA Done: ", cal_hv_front(agea_pop.ParetoFront[0], np.array([10, 100000])))
    return history
