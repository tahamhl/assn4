import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from math import sqrt

# ---------------------------------------------------------
# OR-Tools check (optional)
# ---------------------------------------------------------
USE_ORTOOLS = False
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    USE_ORTOOLS = True
except:
    USE_ORTOOLS = False


# =========================================================
# Distance (Euclidean)
# =========================================================
def dist(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


# =========================================================
# Generate random topology
# =========================================================
def generate_topology(n_nodes, seed):
    rng = np.random.default_rng(seed)
    return rng.random((n_nodes, 2))


# =========================================================
# Tour length
# =========================================================
def tour_length(points, tour):
    total = 0.0
    for i in range(len(tour)):
        a = tour[i]
        b = tour[(i + 1) % len(tour)]
        total += dist(points[a], points[b])
    return total


# =========================================================
# Nearest Neighbor
# =========================================================
def nearest_neighbor(points):
    n = len(points)
    visited = [False] * n
    tour = [0]
    visited[0] = True

    for _ in range(n - 1):
        last = tour[-1]
        best = None
        best_d = 1e18

        for i in range(n):
            if not visited[i]:
                d = dist(points[last], points[i])
                if d < best_d:
                    best_d = d
                    best = i

        tour.append(best)
        visited[best] = True

    return tour


# =========================================================
# 2-opt
# =========================================================
def two_opt(points, tour):
    improved = True
    best = tour[:]
    best_len = tour_length(points, best)
    n = len(tour)

    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                new_tour = best[:]
                new_tour[i:j] = reversed(new_tour[i:j])
                new_len = tour_length(points, new_tour)

                if new_len < best_len:
                    best = new_tour
                    best_len = new_len
                    improved = True

    return best


def heuristic_solver(points):
    tour = nearest_neighbor(points)
    tour = two_opt(points, tour)
    return tour


# =========================================================
# Simulated Annealing
# =========================================================
def simulated_annealing(points, iterations=2000):
    n = len(points)
    tour = np.arange(n)
    np.random.shuffle(tour)

    best = tour.copy()
    best_len = tour_length(points, best)

    T = 1.0

    for _ in range(iterations):
        i, j = sorted(np.random.choice(n, 2, replace=False))
        new = tour.copy()
        new[i:j] = new[i:j][::-1]

        Lnew = tour_length(points, new)

        if Lnew < best_len or np.random.rand() < np.exp((best_len - Lnew) / T):
            tour = new
            if Lnew < best_len:
                best = new
                best_len = Lnew

        T *= 0.9995

    return best


# =========================================================
# Genetic Algorithm (AI SOLVER)
# =========================================================
def genetic_algorithm_solver(
    points,
    pop_size=80,
    generations=300,
    mutation_rate=0.15,
    elite_size=5
):
    n = len(points)

    def create_individual():
        ind = np.arange(n)
        np.random.shuffle(ind)
        return ind

    def fitness(ind):
        return 1.0 / tour_length(points, ind)

    def crossover(p1, p2):
        a, b = sorted(np.random.choice(n, 2, replace=False))
        child = [-1] * n
        child[a:b] = p1[a:b]

        ptr = 0
        for x in p2:
            if x not in child:
                while child[ptr] != -1:
                    ptr += 1
                child[ptr] = x

        return np.array(child)

    def mutate(ind):
        if np.random.rand() < mutation_rate:
            i, j = np.random.choice(n, 2, replace=False)
            ind[i], ind[j] = ind[j], ind[i]

    # Initial population
    population = [create_individual() for _ in range(pop_size)]

    for _ in range(generations):
        population = sorted(population, key=lambda x: tour_length(points, x))
        new_pop = population[:elite_size]

        fitness_vals = np.array([fitness(ind) for ind in population])
        probs = fitness_vals / fitness_vals.sum()

        while len(new_pop) < pop_size:
            p1, p2 = population[
                np.random.choice(len(population), 2, p=probs)
            ]
            child = crossover(p1, p2)
            mutate(child)
            new_pop.append(child)

        population = new_pop

    best = min(population, key=lambda x: tour_length(points, x))
    return best


# =========================================================
# OR-Tools solver (optional baseline)
# =========================================================
def solve_ortools(points):
    n = len(points)
    dist_matrix = [[dist(points[i], points[j]) for j in range(n)] for i in range(n)]

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def cb(i, j):
        a = manager.IndexToNode(i)
        b = manager.IndexToNode(j)
        return int(dist_matrix[a][b] * 1000)

    transit = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 2

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return None

    idx = routing.Start(0)
    tour = []
    while not routing.IsEnd(idx):
        tour.append(manager.IndexToNode(idx))
        idx = solution.Value(routing.NextVar(idx))

    return tour


# =========================================================
# Experiment Runner (30 instances)
# =========================================================
def run_experiment(
    n_topologies=30,
    n_nodes=50,
    base_seed=12345,
    output_dir="tsp_assignment4_output"
):
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for t in range(n_topologies):
        seed = base_seed + t
        points = generate_topology(n_nodes, seed)

        # Heuristic
        t0 = time.time()
        h_tour = heuristic_solver(points)
        h_len = tour_length(points, h_tour)
        h_rt = time.time() - t0
        rows.append(["Heuristic", t, h_len, h_rt, seed])

        # Simulated Annealing
        t0 = time.time()
        sa_tour = simulated_annealing(points)
        sa_tour = two_opt(points, sa_tour)
        sa_len = tour_length(points, sa_tour)
        sa_rt = time.time() - t0
        rows.append(["SA", t, sa_len, sa_rt, seed])

        # Genetic Algorithm (AI)
        t0 = time.time()
        ga_tour = genetic_algorithm_solver(points)
        ga_len = tour_length(points, ga_tour)
        ga_rt = time.time() - t0
        rows.append(["GeneticAlgorithm", t, ga_len, ga_rt, seed])

        # OR-Tools (optional)
        if USE_ORTOOLS:
            t0 = time.time()
            o_tour = solve_ortools(points)
            o_rt = time.time() - t0
            if o_tour is not None:
                o_len = tour_length(points, o_tour)
                rows.append(["OR-Tools", t, o_len, o_rt, seed])

    # Save results
    df = pd.DataFrame(rows, columns=[
        "method", "topology", "tour_length", "runtime", "seed"
    ])
    df.to_csv(f"{output_dir}/results.csv", index=False)

    summary = df.groupby("method")[["tour_length", "runtime"]].agg(["mean", "std"])
    summary.to_csv(f"{output_dir}/summary.csv")

    # -----------------------------------------------------
    # Plots
    # -----------------------------------------------------
    plt.figure()
    df.boxplot(column="tour_length", by="method")
    plt.title("Tour Length Comparison")
    plt.suptitle("")
    plt.ylabel("Tour Length")
    plt.savefig(f"{output_dir}/tour_length_boxplot.png", dpi=300)
    plt.close()

    plt.figure()
    for m in df["method"].unique():
        d = df[df["method"] == m]
        plt.scatter(d["runtime"], d["tour_length"], label=m)
    plt.xlabel("Runtime (s)")
    plt.ylabel("Tour Length")
    plt.legend()
    plt.savefig(f"{output_dir}/runtime_vs_length.png", dpi=300)
    plt.close()

    # Combined plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df.boxplot(column="tour_length", by="method", ax=axes[0])
    axes[0].set_title("Tour Length")
    axes[0].set_xlabel("")

    for m in df["method"].unique():
        d = df[df["method"] == m]
        axes[1].scatter(d["runtime"], d["tour_length"], label=m)
    axes[1].set_xlabel("Runtime (s)")
    axes[1].set_ylabel("Tour Length")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_plots.png", dpi=300)
    plt.close()

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------
    with open(f"{output_dir}/comparison_report.txt", "w") as f:
        f.write("=== Assignment 4: AI Technique Integration ===\n")
        f.write("AI Method: Genetic Algorithm\n")
        f.write(f"OR-Tools available: {USE_ORTOOLS}\n\n")
        f.write(summary.to_string())

    print("Experiment finished.")
    print("Results saved in:", output_dir)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    run_experiment()
