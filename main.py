import random
import tkinter as tk
import time
from copy import deepcopy
import matplotlib.pyplot as plt

# Constants to avoid magic numbers
SMALL_POP_THRESHOLD = 5
LARGE_POP_THRESHOLD = 30

SMALL_POP_FLIPS_MIN = 5
SMALL_POP_FLIPS_MAX = 10

LARGE_POP_FLIPS_MIN = 15
LARGE_POP_FLIPS_MAX = 20

MODERATE_POP_FLIPS_MIN = 1
MODERATE_POP_FLIPS_MAX = 6

TOURNAMENT_SIZE = 3
CROSSOVER_MARGIN = 3

WARNING_ATTEMPTS_FACTOR = 3

# Game of Life Simulation
def count_neighbors(grid):
    rows = len(grid)
    cols = len(grid[0])
    neighbors = [[0 for _ in range(cols)] for _ in range(rows)]

    for row in range(rows):
        for col in range(cols):
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_row = row + dx
                    new_col = col + dy
                    if 0 <= new_row < rows and 0 <= new_col < cols:
                        neighbors[row][col] += grid[new_row][new_col]
    return neighbors

def update_grid(grid):
    rows = len(grid)
    cols = len(grid[0])
    neighbors = count_neighbors(grid)
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]

    for row in range(rows):
        for col in range(cols):
            cell = grid[row][col]
            neighbor_count = neighbors[row][col]
            if cell == 1 and (neighbor_count == 2 or neighbor_count == 3):
                new_grid[row][col] = 1
            elif cell == 0 and neighbor_count == 3:
                new_grid[row][col] = 1
    return new_grid

def get_fitness(grid, max_steps):
    steps = 0
    current_grid = [row[:] for row in grid]
    initial_population = sum(sum(row) for row in current_grid)
    if initial_population == 0:
        # No cells initially: no fitness
        return 0, 0, 1, 0

    max_population = initial_population
    grid_history = [current_grid]

    while steps < max_steps:
        grid_tuple = tuple(tuple(row) for row in current_grid)

        # Detect period
        for period_length in range(1, len(grid_history)):
            if grid_tuple == tuple(tuple(r) for r in grid_history[-period_length]):
                final_population = sum(sum(r) for r in current_grid)
                fitness = (steps * steps) / (initial_population * initial_population * (final_population + 1))
                return fitness, steps, period_length, max_population

        grid_history.append(current_grid)
        next_grid = update_grid(current_grid)
        current_population = sum(sum(r) for r in next_grid)

        if current_population == 0:
            # Extinction
            fitness = (steps * steps) / (initial_population * initial_population)
            return fitness, steps, 1, max_population

        max_population = max(max_population, current_population)
        current_grid = next_grid
        steps += 1

    final_population = sum(sum(r) for r in current_grid)
    fitness = (steps * steps) / (initial_population * initial_population * (final_population + 1))
    return fitness, steps, None, max_population

def random_cell_flips(grid, mutation_chance):
    if random.random() < mutation_chance:
        pop = sum(sum(row) for row in grid)
        rows = len(grid)
        cols = len(grid[0])
        neighbors = count_neighbors(grid)

        # Determine flips range based on population
        if pop <= SMALL_POP_THRESHOLD:
            dynamic_min_flips = SMALL_POP_FLIPS_MIN
            dynamic_max_flips = SMALL_POP_FLIPS_MAX
        elif pop >= LARGE_POP_THRESHOLD:
            dynamic_min_flips = LARGE_POP_FLIPS_MIN
            dynamic_max_flips = LARGE_POP_FLIPS_MAX
        else:
            dynamic_min_flips = MODERATE_POP_FLIPS_MIN
            dynamic_max_flips = MODERATE_POP_FLIPS_MAX

        # Determine probability for value = 1
        if pop <= SMALL_POP_THRESHOLD:
            prob_1 = 0
        elif pop >= LARGE_POP_THRESHOLD:
            prob_1 = 1
        else:
            # Scale probability linearly between pop=5 (prob_1=0) and pop=30 (prob_1=1)
            prob_1 = (pop - SMALL_POP_THRESHOLD) / (LARGE_POP_THRESHOLD - SMALL_POP_THRESHOLD)

        prob_0 = 1 - prob_1
        value = random.choices([1, 0], weights=[prob_1, prob_0])[0]

        candidate_cells = [(r, c) for r in range(rows) for c in range(cols)
                           if neighbors[r][c] >= 2 and grid[r][c] == value]

        if candidate_cells:
            flips = random.randint(dynamic_min_flips, dynamic_max_flips)
            flips = min(flips, len(candidate_cells))
            cells_to_flip = random.sample(candidate_cells, flips)
            for (r, c) in cells_to_flip:
                grid[r][c] = 1 - grid[r][c]

    return grid

def create_random_grid(size, max_live_cells):
    grid = [[0 for _ in range(size)] for _ in range(size)]
    start_row = size//2
    start_col = size//2

    live_cells = 0
    attempts = 0
    max_attempts = max_live_cells * WARNING_ATTEMPTS_FACTOR

    while attempts < max_attempts and live_cells < max_live_cells:
        row_offset = random.randint(-2, 2)
        col_offset = random.randint(-2, 2)
        new_row = start_row + row_offset
        new_col = start_col + col_offset

        if 0 <= new_row < size and 0 <= new_col < size and grid[new_row][new_col] == 0:
            grid[new_row][new_col] = 1
            live_cells += 1
        attempts += 1

    if live_cells < max_live_cells:
        print(f"Warning: Could only place {live_cells}/{max_live_cells} live cells.")

    return grid

def create_population(population_size, grid_size, max_live_cells):
    print(f"Creating population of size {population_size}...")
    population = []
    for i in range(population_size):
        g = create_random_grid(grid_size, max_live_cells)
        population.append(g)
        placed_cells = sum(sum(row) for row in g)
        print(f"Grid {i + 1}: {placed_cells} live cells placed.")
    print("Population created.")
    return population

def crossover(parent1, parent2):
    size = len(parent1)
    split = random.randint(CROSSOVER_MARGIN, size - CROSSOVER_MARGIN)
    return parent1[:split] + parent2[split:]

def mutate(grid, mutation_chance):
    return random_cell_flips(grid, mutation_chance)

def evolve_population(population, fitness_scores, mutation_chance):
    print("Evolving population...")
    sorted_pairs = sorted(zip(fitness_scores, population), reverse=True)
    sorted_population = [g for _, g in sorted_pairs]
    sorted_fitness_scores = [f for f, _ in sorted_pairs]

    if not sorted_population:
        print("No viable grids left to evolve. Stopping evolution.")
        return []

    new_population = []
    new_population.append(sorted_population[0])
    while len(new_population) < len(population):
        parent1 = tournament_selection(sorted_population, sorted_fitness_scores, TOURNAMENT_SIZE)
        parent2 = tournament_selection(sorted_population, sorted_fitness_scores, TOURNAMENT_SIZE)
        child = crossover(parent1, parent2)
        child = mutate(child, mutation_chance)
        new_population.append(child)

    print("Population evolved.")
    return new_population

def find_best_grid(grid_size, population_size, generations, base_mutation_chance, max_steps, max_live_cells):
    population = create_population(population_size, grid_size, max_live_cells)

    global_best_fitness = 0
    global_best_grid = None
    global_best_steps = 0
    global_best_period = None
    global_best_max_population = 0

    fitness_history = []
    previous_generation_fitness = None


    for gen in range(generations):
        print(f"Generation {gen + 1}/{generations}")
        fitness_scores = []
        actual_steps_per_grid = []
        actual_periods_per_grid = []
        actual_max_pops_per_grid = []

        for g in population:
            fitness, steps, period, max_pop = get_fitness(g, max_steps)
            fitness_scores.append(fitness)
            actual_steps_per_grid.append(steps)
            actual_periods_per_grid.append(period)
            actual_max_pops_per_grid.append(max_pop)

        max_fitness_in_gen = max(fitness_scores)
        fitness_history.append(max_fitness_in_gen)

        if max_fitness_in_gen > global_best_fitness:
            global_best_fitness = max_fitness_in_gen
            best_index = fitness_scores.index(max_fitness_in_gen)
            global_best_grid = deepcopy(population[best_index])
            global_best_steps = actual_steps_per_grid[best_index]
            global_best_period = actual_periods_per_grid[best_index]
            global_best_max_population = actual_max_pops_per_grid[best_index]

        print(f"Best fitness in generation {gen + 1}: {max_fitness_in_gen}")

        # Adjust mutation rate based on stagnation
        if (previous_generation_fitness is not None and
                (max_fitness_in_gen == previous_generation_fitness or max_fitness_in_gen * 10 < global_best_fitness)):
            mutation_chance = 0.97
        else:
            mutation_chance = 0.1

        previous_generation_fitness = max_fitness_in_gen

        if not any(fitness_scores):
            print("All grids died or no periodic/stable solution found.")
            break

        population = evolve_population(population, fitness_scores, mutation_chance)
        if not population:
            print("No more evolution possible.")
            break

    print(f"Best fitness overall (all generations): {global_best_fitness}")
    return global_best_grid, global_best_fitness, global_best_steps, global_best_period, global_best_max_population, fitness_history

class GameOfLifeApp(tk.Tk):
    def __init__(self, grid, steps, max_population, population_size, final_period, delay=0.2):
        super().__init__()
        self.grid = grid
        self.steps = steps
        self.max_population = max_population
        self.population_size = population_size
        self.final_period = final_period
        self.delay = delay
        self.grid_size = len(grid)
        self.cell_size = 20

        self.title("Game of Life - Best Fitness Simulation")
        self.geometry(f"{self.grid_size * self.cell_size}x{self.grid_size * self.cell_size + 150}")

        self.canvas = tk.Canvas(self, width=self.grid_size * self.cell_size,
                                height=self.grid_size * self.cell_size, bg="white")
        self.canvas.pack()

        self.control_frame = tk.Frame(self)
        self.control_frame.pack()

        self.start_button = tk.Button(self.control_frame, text="Start", command=self.run_simulation)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.step_label = tk.Label(self.control_frame, text="Step: 0")
        self.step_label.pack(side=tk.LEFT)

        self.info_frame = tk.Frame(self)
        self.info_frame.pack()

        self.max_pop_label = tk.Label(self.info_frame, text=f"Max Cells Alive: {self.max_population}")
        self.max_pop_label.pack(side=tk.LEFT, padx=10)

        self.total_steps_label = tk.Label(self.info_frame, text=f"Total Steps: {self.steps}")
        self.total_steps_label.pack(side=tk.LEFT, padx=10)

        self.final_period_label = tk.Label(self.info_frame, text=f"Final Period: {self.final_period}")
        self.final_period_label.pack(side=tk.LEFT, padx=10)

        self.draw_grid(self.grid)

    def draw_grid(self, grid):
        self.canvas.delete("all")
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                fill_color = "black" if grid[row][col] == 1 else "white"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="black")

    def run_simulation(self):
        grid = [row[:] for row in self.grid]
        self.draw_grid(grid)
        self.update()

        max_steps = self.steps
        original_states = [grid]
        period_detected = False

        for step in range(max_steps):
            time.sleep(self.delay)
            grid = update_grid(grid)
            self.draw_grid(grid)
            self.update()

            for period_length in range(1, len(original_states)):
                if grid == original_states[-period_length]:
                    print(f"Period of {period_length} detected. Stopping simulation.")
                    period_detected = True
                    break

            if period_detected:
                break

            original_states.append(grid)
            self.step_label.config(text=f"Step: {step + 1}")
            self.update()

            if sum(sum(r) for r in grid) == 0:
                print("Empty grid detected, considered stable with period=1. Stopping simulation.")
                break

        print(f"Simulation stopped after {step + 1} steps")

def tournament_selection(population, fitness_scores, tournament_size):
    participants = random.sample(list(zip(population, fitness_scores)), tournament_size)
    winner = max(participants, key=lambda x: x[1])[0]
    return winner

if __name__ == "__main__":
    GRID_SIZE = 25
    POPULATION_SIZE = 75
    GENERATIONS = 20
    BASE_MUTATION_CHANCE = 0.1
    MAX_STEPS = 500
    MAX_LIVE_CELLS = 5

    print("Starting genetic algorithm...")
    best_grid, best_fitness, best_steps, best_period, best_max_population, fitness_history = find_best_grid(
        GRID_SIZE, POPULATION_SIZE, GENERATIONS,
        BASE_MUTATION_CHANCE, MAX_STEPS, MAX_LIVE_CELLS
    )
    print("Genetic algorithm finished.")

    if fitness_history:
        plt.plot(range(1, len(fitness_history) + 1), fitness_history)
        plt.title("Fitness Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.show()

    if best_grid is not None and (best_period is not None or best_period == 1):
        print(f"Displaying the best solution for {best_steps} steps...")
        print(f"Best Period: {best_period}")
        app = GameOfLifeApp(best_grid, steps=best_steps, max_population=best_max_population,
                            population_size=POPULATION_SIZE, final_period=best_period, delay=0.05)
        app.mainloop()
    else:
        print("No viable (periodic or stable) solution found.")
