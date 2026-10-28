Claude























Studentid lab2 · PY
"""
BSD3513 Introduction to Artificial Intelligence - Lab Report 2
Chapter 2: Search Algorithms (Genetic Algorithm)
 
Problem
-------
Evolve a bit pattern with a Genetic Algorithm (GA) where
    * population size            = 300
    * individual (chromosome) length = 80 bits
    * generations                = 50
    * fitness is maximal (= 80) when the number of ones equals 50:
          fitness(x) = LENGTH - |TARGET_ONES - sum(x)|
 
Run locally :  pip install -r requirements.txt
               streamlit run StudentID_Lab2.py
"""
import numpy as np
import pandas as pd
import streamlit as st
 
# ----------------------------------------------------------------------------
# 1. Problem definition (values required by the lab sheet)
# ----------------------------------------------------------------------------
POP_SIZE = 300
LENGTH = 80
TARGET_ONES = 50
GENERATIONS = 50
 
 
# ----------------------------------------------------------------------------
# 2. Genetic-algorithm building blocks
# ----------------------------------------------------------------------------
def fitness(pop: np.ndarray, length: int, target: int) -> np.ndarray:
    """Fitness of every row (individual). Max = `length` when #ones == target."""
    return length - np.abs(target - pop.sum(axis=1))
 
 
def init_population(rng, pop_size, length, p_one=0.5):
    """Random bit-strings; each gene is 1 with probability p_one."""
    return (rng.random((pop_size, length)) < p_one).astype(np.int8)
 
 
def tournament_selection(pop, fit, k, rng):
    """Pick k random individuals per slot and keep the fittest (selection operator)."""
    n = len(pop)
    contenders = rng.integers(0, n, size=(n, k))
    winners = contenders[np.arange(n), fit[contenders].argmax(axis=1)]
    return pop[winners].copy()
 
 
def one_point_crossover(parents, p_cross, rng):
    """Pair up neighbours; with prob. p_cross swap their tails after a random point."""
    children = parents.copy()
    n, length = parents.shape
    for i in range(0, n - 1, 2):
        if rng.random() < p_cross:
            cut = rng.integers(1, length)
            tail_a = parents[i, cut:].copy()
            children[i, cut:] = parents[i + 1, cut:]
            children[i + 1, cut:] = tail_a
    return children
 
 
def bit_flip_mutation(children, p_mut, rng):
    """Flip every bit independently with probability p_mut (diversity)."""
    flips = rng.random(children.shape) < p_mut
    return np.where(flips, 1 - children, children).astype(np.int8)
 
 
def run_ga(pop_size=POP_SIZE, length=LENGTH, target=TARGET_ONES,
           generations=GENERATIONS, p_cross=0.8, p_mut=0.01,
           tournament_k=3, elite=1, p_one=0.5, seed=42):
    """
    Full GA loop:  initialise -> [evaluate -> select -> crossover -> mutate -> replace]
    Returns (history DataFrame, final population, best individual).
    Row 0 of the history is the random initial population (generation 0).
    """
    rng = np.random.default_rng(seed)
    pop = init_population(rng, pop_size, length, p_one)
    rows = []
 
    for gen in range(generations + 1):
        fit = fitness(pop, length, target)
        best_i = int(fit.argmax())
        rows.append({
            "generation": gen,
            "best_fitness": int(fit.max()),
            "mean_fitness": float(fit.mean()),
            "worst_fitness": int(fit.min()),
            "best_ones": int(pop[best_i].sum()),
            "mean_ones": float(pop.sum(axis=1).mean()),
            "n_optimal": int((fit == length).sum()),
        })
        if gen == generations:
            break
        elites = pop[np.argsort(fit)[::-1][:elite]].copy()      # elitism
        parents = tournament_selection(pop, fit, tournament_k, rng)
        children = one_point_crossover(parents, p_cross, rng)
        children = bit_flip_mutation(children, p_mut, rng)
        children[:elite] = elites                                # keep best
        pop = children
 
    fit = fitness(pop, length, target)
    return pd.DataFrame(rows), pop, pop[int(fit.argmax())]
 
 
def trace_one_generation(seed=7, n=6, length=10, target=5, k=2, p_cross=1.0, p_mut=0.1):
    """Tiny worked example (6 individuals x 10 bits) so every step can be inspected."""
    rng = np.random.default_rng(seed)
    pop = init_population(rng, n, length)
    fit = fitness(pop, length, target)
    parents = tournament_selection(pop, fit, k, rng)
    children = one_point_crossover(parents, p_cross, rng)
    mutated = bit_flip_mutation(children, p_mut, rng)
    fit_new = fitness(mutated, length, target)
    s = lambda a: ["".join(map(str, r)) for r in a]
    table = pd.DataFrame({
        "1) Initial": s(pop), "ones": pop.sum(1), "fitness": fit,
        "2) Selected parent": s(parents),
        "3) After crossover": s(children),
        "4) After mutation": s(mutated), "ones (new)": mutated.sum(1),
        "fitness (new)": fit_new,
    })
    table.index = [f"I{i + 1}" for i in range(n)]
    return table, length, target
 
 
# ----------------------------------------------------------------------------
# 3. Streamlit user interface
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="BSD3513 Lab 2 - Genetic Algorithm", layout="wide")
    st.title("Genetic Algorithm - Bit-Pattern Evolution")
    st.caption("BSD3513 Introduction to Artificial Intelligence | Lab Report 2 | Chapter 2: Search Algorithms")
 
    tab_ai, tab_run, tab_trace, tab_cx = st.tabs([
        "AI types & GA applications", "GA simulation (Q2)",
        "Step-by-step trace", "Complexity"])
 
    # ---- Tab 1: general knowledge ------------------------------------------
    with tab_ai:
        st.subheader("AI capability types and where they are deployed")
        st.table(pd.DataFrame({
            "Type": ["Reactive machines", "Limited memory", "Theory of mind", "Self-aware AI",
                     "Narrow AI (ANI)", "General AI (AGI)", "Super AI (ASI)"],
            "Meaning": ["Respond to current input only, no memory",
                        "Use recent/historical data to decide",
                        "Would model beliefs/intentions of others (research stage)",
                        "Would have its own consciousness (hypothetical)",
                        "One task only",
                        "Human-level across tasks (not yet achieved)",
                        "Beyond human ability (hypothetical)"],
            "Real deployment example": ["Chess engines (e.g. Deep Blue)",
                                        "Self-driving perception, recommender systems, GA/ML models",
                                        "Research on social robots",
                                        "None",
                                        "Spam filters, route optimisers, a GA that tunes a design",
                                        "None in production",
                                        "None"],
        }))
        st.info("A genetic algorithm is a **narrow-AI** heuristic search/optimisation technique: it is "
                "deployed for one well-defined objective (a fitness function), not for general reasoning.")
        st.subheader("Three real-world applications of genetic algorithms")
        st.markdown(
            "1. **Engineering design** - evolved antenna shapes for NASA's ST5 mission (Hornby et al., 2006).\n"
            "2. **Logistics / routing & scheduling** - vehicle-routing and timetabling (Prins, 2004).\n"
            "3. **Machine learning** - feature selection and evolving neural networks (Yang & Honavar, 1998; "
            "Stanley & Miikkulainen, 2002).\n\n"
            "Full discussion and reference list are in the PDF report.")
 
    # ---- Tab 2: simulation -------------------------------------------------
    with tab_run:
        st.subheader("Run the GA with the lab-sheet parameters")
        with st.sidebar:
            st.header("Parameters")
            st.caption("Defaults = values required by the lab sheet.")
            pop_size = st.slider("Population size", 50, 1000, POP_SIZE, 10)
            length = st.slider("Individual length (bits)", 20, 200, LENGTH, 1)
            target = st.slider("Target number of ones", 0, length, min(TARGET_ONES, length), 1)
            gens = st.slider("Generations", 5, 200, GENERATIONS, 1)
            st.divider()
            p_cross = st.slider("Crossover probability", 0.0, 1.0, 0.8, 0.05)
            p_mut = st.slider("Mutation probability (per bit)", 0.0, 0.2, 0.01, 0.005)
            k = st.slider("Tournament size", 2, 10, 3, 1)
            elite = st.slider("Elites kept each generation", 0, 10, 1, 1)
            p_one = st.slider("Initial P(bit = 1)", 0.05, 0.95, 0.5, 0.05,
                              help="0.5 = uniform random start. Lower it to see a clearer evolution curve.")
            seed = st.number_input("Random seed", 0, 10_000, 42, 1)
 
        hist, final_pop, best = run_ga(pop_size, length, target, gens, p_cross, p_mut,
                                       k, elite, p_one, int(seed))
        max_fit = length
        hit = hist.index[hist.best_fitness == max_fit]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best fitness (final)", f"{hist.best_fitness.iloc[-1]} / {max_fit}")
        c2.metric("Ones in best individual", int(best.sum()), f"target {target}", delta_color="off")
        c3.metric("First generation at optimum", int(hit[0]) if len(hit) else "not reached")
        c4.metric("Mean fitness (final)", f"{hist.mean_fitness.iloc[-1]:.2f}")
 
        st.markdown("**Fitness over generations**")
        st.line_chart(hist.set_index("generation")[["best_fitness", "mean_fitness", "worst_fitness"]])
        st.markdown("**Average number of ones in the population** (should approach the target)")
        st.line_chart(hist.set_index("generation")[["mean_ones"]])
        st.markdown("**Best bit pattern found**")
        st.code("".join(map(str, best)) + f"\nones = {int(best.sum())}, fitness = {int(fitness(best[None, :], length, target)[0])}")
        with st.expander("Generation-by-generation table"):
            st.dataframe(hist)
        st.download_button("Download history (CSV)", hist.to_csv(index=False).encode(),
                           "ga_history.csv", "text/csv")
 
    # ---- Tab 3: trace ------------------------------------------------------
    with tab_trace:
        st.subheader("Trace of one generation (6 individuals, 10 bits, target = 5 ones)")
        st.write("Fitness = 10 - |5 - ones|. Tournament size 2, crossover probability 1, "
                 "mutation probability 0.1 per bit so every operator is visible.")
        seed_t = st.number_input("Trace seed", 0, 1000, 7, 1)
        table, ln, tg = trace_one_generation(seed=int(seed_t))
        st.dataframe(table)
        st.markdown(
            "- **Selection** copies fitter individuals more often (a parent may appear twice).\n"
            "- **Crossover** swaps the tails of neighbouring parents after a random cut point.\n"
            "- **Mutation** flips a few bits to keep diversity and avoid premature convergence.\n"
            f"- Mean fitness: **{table['fitness'].mean():.2f} -> {table['fitness (new)'].mean():.2f}**")
 
    # ---- Tab 4: complexity -------------------------------------------------
    with tab_cx:
        st.subheader("Time and space complexity")
        P, L, G = POP_SIZE, LENGTH, GENERATIONS
        st.markdown(
            "Let **P** = population size, **L** = chromosome length, **G** = generations, "
            "**k** = tournament size.\n\n"
            "| Operator | Cost per generation |\n|---|---|\n"
            "| Fitness evaluation (sum of bits) | O(P×L) |\n"
            "| Tournament selection | O(P×k) |\n"
            "| One-point crossover | O(P×L) |\n"
            "| Bit-flip mutation | O(P×L) |\n\n"
            "**Time:** O(G×P×L) (k is a small constant).  **Space:** O(P×L).")
        a, b, c = st.columns(3)
        a.metric("Fitness evaluations (G×P)", f"{G * P:,}")
        b.metric("Bit operations (G×P×L)", f"{G * P * L:,}")
        c.metric("Brute-force search space 2^L", f"{2 ** L:.2e}")
        st.caption("The GA inspects ~10^4 candidates instead of 1.2 x 10^24, because selection pressure "
                   "steers the search - but unlike BFS/UCS/A*, it does not guarantee the optimum.")
 
 
if __name__ == "__main__":
    main()
 
Claude finished the response
