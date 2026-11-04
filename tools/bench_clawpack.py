"""Work-precision benchmark: fronttrack against Clawpack's finite-volume solvers.

Every method is refined until one run takes longer than --budget seconds. Each
run is scored by its wall time (setup and evolution, no output) and by its L1
error at the final time, measured on the same 2^22 midpoint sample for every
method. Competitors, all from Clawpack's PyClaw with Fortran kernels:

* classic_godunov   ClawSolver1D, first order (Godunov / upwind)
* classic_mc        ClawSolver1D, second order with the MC limiter
* sharpclaw_weno5   SharpClawSolver1D, fifth-order WENO with SSP Runge-Kutta

Problems (f is the flux, S1 the unit circle):

* burgers_box    f = u^2/2, u0 = 1 on [0, 1), T = 8          exact solution
* burgers_sine   f = u^2/2, u0 = sin 2 pi x on S1, T = 1     exact solution
* cubic_riemann  f = u^3, u0 = 1 | -1, T = 1/2               exact solution
* cubic_sine     f = u^3, u0 = sin 2 pi x on S1, T = 1       fine reference

cubic_sine has a nonconvex flux and many interactions, and no closed form. Its
reference is fronttrack at --reference-level (delta = 2^-level), at least
eight times finer than any scored fronttrack run. Clawpack's Godunov scheme is
monotone, so it converges to the entropy solution on its own; its observed
order of about 1 against this reference is the independent check.

Usage (Clawpack needs a Fortran compiler: e.g. apt install gfortran, then
pip install meson ninja clawpack):

    python tools/bench_clawpack.py                       # everything, ~1 hour
    python tools/bench_clawpack.py --problems cubic_sine --budget 5
    python tools/bench_clawpack.py --summarize docs/benchmarks/clawpack.json

Results go to docs/benchmarks/clawpack.json and a plot next to it. Timings are
machine dependent: compare methods within one file, not across machines.
"""
from pathlib import Path
import argparse
import datetime
import json
import math
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fronttrack as ft  # noqa: E402

SAMPLES = 2**22
CLAWPACK = ("classic_godunov", "classic_mc", "sharpclaw_weno5")
LABELS = {"fronttrack": "fronttrack", "classic_godunov": "Clawpack Godunov",
          "classic_mc": "Clawpack classic MC", "sharpclaw_weno5": "SharpClaw WENO5"}


# ------------------------------------------------------------- exact solutions
def box_exact(x, T):
    """Burgers, u0 = 1 on [0, 1): for T >= 2 a fan x/T up to the shock at sqrt(2T)."""
    s = math.sqrt(2 * T)
    return np.where((x > 0) & (x < s), x / T, 0.0)


def sine_exact(x, T):
    """Burgers, u0 = sin 2 pi x on S1: by symmetry the shock sits at x = 1/2.

    On [0, 1/2) the solution is sin(2 pi xi) with x = xi + T sin(2 pi xi),
    and x(xi) increases up to xi_c, the foot of the shock; solve by bisection.
    """
    x = np.mod(x, 1.0)
    left = x < 0.5
    xl = np.where(left, x, 1.0 - x)
    xi_c = math.acos(-1 / (2 * math.pi * T)) / (2 * math.pi) if 2 * math.pi * T > 1 else 0.5
    lo, hi = np.zeros_like(xl), np.full_like(xl, xi_c)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        big = mid + T * np.sin(2 * np.pi * mid) > xl
        hi, lo = np.where(big, mid, hi), np.where(big, lo, mid)
    u = np.sin(np.pi * (lo + hi))
    return np.where(left, u, -u)


def cubic_riemann_exact(x, T):
    """f = u^3, 1 | -1: shock from 1 to -1/2 at speed 3/4, then a fan to -1."""
    r = x / T
    fan = -np.sqrt(np.clip(r, 0.0, None) / 3)
    return np.where(r < 0.75, 1.0, np.where(r < 3.0, fan, -1.0))


def sine(x):
    return np.sin(2 * np.pi * x)


PROBLEMS = {
    "burgers_box": dict(title="Burgers, box data, T = 8", a=-1.0, b=6.0, T=8.0,
                        periodic=False, flux="burgers", u_range=(0.0, 1.0),
                        exact=box_exact, steps=([0.0, 1.0], ["lo", "hi", "lo"]),
                        u0=lambda x: np.where((x >= 0) & (x < 1), 1.0, 0.0)),
    "burgers_sine": dict(title="Burgers, sin 2πx on the circle, T = 1", a=0.0, b=1.0,
                         T=1.0, periodic=True, flux="burgers", u_range=(-1.0, 1.0),
                         exact=sine_exact, u0=sine),
    "cubic_riemann": dict(title="f = u³, Riemann 1 | −1, T = 1/2", a=-0.5, b=2.0, T=0.5,
                          periodic=False, flux="cubic", u_range=(-1.0, 1.0),
                          exact=cubic_riemann_exact, steps=([0.0], ["hi", "lo"]),
                          u0=lambda x: np.where(x < 0, 1.0, -1.0)),
    "cubic_sine": dict(title="f = u³, sin 2πx on the circle, T = 1", a=0.0, b=1.0,
                       T=1.0, periodic=True, flux="cubic", u_range=(-1.0, 1.0),
                       exact=None, u0=sine),
}
FLUXES = {"burgers": lambda u: u * u / 2, "cubic": lambda u: u**3}


def sample_points(p):
    return p["a"] + (np.arange(SAMPLES) + 0.5) * (p["b"] - p["a"]) / SAMPLES


def l1(p, values, truth):
    return float(np.mean(np.abs(values - truth)) * (p["b"] - p["a"]))


# ----------------------------------------------------------------- fronttrack
def fronttrack_solution(p, k):
    """Run fronttrack with delta = 2^-k; return (tracker, wall time)."""
    lo, hi = p["u_range"]
    n = int(round((hi - lo) * 2**k)) + 1
    start = time.perf_counter()
    flux = ft.DiscreteFlux(FLUXES[p["flux"]], np.linspace(lo, hi, n))
    if p["periodic"]:
        # Midpoint data on 2^k cells, rounded to the flux grid.
        tracker = ft.PeriodicFrontTracker.from_function(flux, p["u0"], cells=2**k,
                                                        max_events=10**9)
    else:
        positions, ends = p["steps"]
        states = [0 if e == "lo" else n - 1 for e in ends]
        tracker = ft.FrontTracker(flux, positions, states, record_history=False,
                                  max_events=10**9)
    tracker.run(p["T"])
    return tracker, time.perf_counter() - start


def run_fronttrack(p, k, x, truth):
    tracker, seconds = fronttrack_solution(p, k)
    return dict(param=2.0**-k, label=f"δ=2^-{k}", time=seconds,
                err=l1(p, tracker.sample(x), truth),
                work=dict(events=tracker.event_count, fronts=len(tracker.fronts)))


# ------------------------------------------------------------------- clawpack
def run_clawpack(kind, p, N, x, truth):
    from clawpack import pyclaw, riemann
    rp = riemann.burgers_1D if p["flux"] == "burgers" else riemann.cubic_1D
    start = time.perf_counter()
    if kind == "sharpclaw_weno5":
        solver = pyclaw.SharpClawSolver1D(rp)
        solver.weno_order = 5
    else:
        solver = pyclaw.ClawSolver1D(rp)
        solver.order = 1 if kind == "classic_godunov" else 2
        solver.limiters = pyclaw.limiters.tvd.MC
        solver.cfl_max, solver.cfl_desired = 1.0, 0.9
    solver.kernel_language = "Fortran"
    solver.max_steps = 10**9  # PyClaw's default (10000) silently stops short of T.
    bc = pyclaw.BC.periodic if p["periodic"] else pyclaw.BC.extrap
    solver.bc_lower[0] = solver.bc_upper[0] = bc
    domain = pyclaw.Domain(pyclaw.Dimension(p["a"], p["b"], N, name="x"))
    state = pyclaw.State(domain, 1)
    state.problem_data["efix"] = True  # Entropy fix (used by the Burgers solver).
    state.q[0, :] = p["u0"](state.grid.x.centers)
    claw = pyclaw.Controller()
    claw.solution, claw.solver = pyclaw.Solution(state, domain), solver
    claw.tfinal, claw.num_output_times = p["T"], 1
    claw.output_format, claw.keep_copy, claw.verbosity = None, True, 0
    claw.run()
    seconds = time.perf_counter() - start
    final = claw.frames[-1]
    if abs(final.t - p["T"]) > 1e-9:
        raise RuntimeError(f"{kind} N={N} stopped at t={final.t}, not T={p['T']}")
    cell = np.minimum(((x - p["a"]) / (p["b"] - p["a"]) * N).astype(int), N - 1)
    return dict(param=(p["b"] - p["a"]) / N, label=f"N={N}", time=seconds,
                err=l1(p, final.q[0][cell], truth),
                work=dict(steps=solver.status["numsteps"]))


# ---------------------------------------------------------------------- sweep
def sweep(name, run, levels, budget):
    rows = []
    print(f"  {name}", flush=True)
    for level in levels:
        row = run(level)
        rows.append(row)
        print(f"    {row['label']:>10}  L1 {row['err']:.3e}  {row['time']:9.4f} s  "
              f"{row['work']}", flush=True)
        if row["time"] > budget:
            break
    return rows


def measure(args):
    names = args.problems or list(PROBLEMS)
    methods = ["fronttrack"] + ([] if args.skip_clawpack else list(CLAWPACK))
    if not args.skip_clawpack:
        try:
            import clawpack  # noqa: F401
        except ImportError:
            sys.exit("Clawpack is not installed; see the docstring, or pass --skip-clawpack.")
    out = dict(meta=meta(args), problems={})
    warm = set()
    for name in names:
        p = PROBLEMS[name]
        print(name, flush=True)
        x = sample_points(p)
        record = dict(title=p["title"], T=p["T"], methods={})
        max_k = args.max_level
        if p["exact"] is not None:
            truth = p["exact"](x, p["T"])
            record["reference"] = "exact"
        else:
            tracker, seconds = fronttrack_solution(p, args.reference_level)
            truth = tracker.sample(x)
            max_k = min(max_k, args.reference_level - 3)
            record["reference"] = dict(method="fronttrack", delta=2.0**-args.reference_level,
                                       time=seconds, events=tracker.event_count)
            print(f"  reference: fronttrack δ=2^-{args.reference_level}, "
                  f"{tracker.event_count} events, {seconds:.1f} s", flush=True)
        for method in methods:
            if method == "fronttrack":
                run = lambda k: run_fronttrack(p, k, x, truth)  # noqa: E731
                levels = range(4, max_k + 1)
            else:
                run = lambda k, m=method: run_clawpack(m, p, 2**k, x, truth)  # noqa: E731
                levels = range(6, 21)
            if method not in warm:
                run(levels[0])  # Untimed: the first call loads modules and kernels.
                warm.add(method)
            record["methods"][method] = sweep(LABELS[method], run, levels, args.budget)
        out["problems"][name] = record
        # Save after every problem so a long run can be inspected or resumed.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1) + "\n")
    return out


def meta(args):
    try:
        from clawpack import __version__ as claw_version
    except ImportError:
        claw_version = None
    return dict(date=datetime.date.today().isoformat(), budget_s=args.budget,
                python=platform.python_version(), numpy=np.__version__,
                fronttrack=ft.__version__, clawpack=claw_version,
                machine=f"{platform.system()} {platform.machine()}", samples=SAMPLES)


# ------------------------------------------------------------------ summarize
TARGETS = (1e-2, 1e-3, 1e-4, 1e-5)


def time_to(rows, target):
    """Wall time to reach `target` L1 error, interpolated in log-log; or None."""
    for i, row in enumerate(rows):
        if row["err"] <= target:
            if i == 0:
                return row["time"]
            e0, e1 = math.log(rows[i - 1]["err"]), math.log(row["err"])
            w = (e0 - math.log(target)) / (e0 - e1)
            return math.exp((1 - w) * math.log(rows[i - 1]["time"]) + w * math.log(row["time"]))
    return None


def observed_order(rows):
    """Least-squares slope of log error against log resolution over the four
    finest runs (coarse runs are still in their pre-asymptotic phase)."""
    use = rows[-4:]
    if len(use) < 2:
        return float("nan")
    return float(np.polyfit(np.log([r["param"] for r in use]),
                            np.log([r["err"] for r in use]), 1)[0])


def summary(results):
    lines = []
    for record in results["problems"].values():
        methods = record["methods"]
        lines += [f"\n### {record['title']}\n",
                  "| method | finest run | L1 error | time (s) | order | "
                  + " | ".join(f"time to {t:.0e}" for t in TARGETS) + " |",
                  "|---|---|---|---|---|" + "---|" * len(TARGETS)]
        for method, rows in methods.items():
            cells = [time_to(rows, t) for t in TARGETS]
            lines.append(f"| {LABELS[method]} | {rows[-1]['label']} | {rows[-1]['err']:.2e} | "
                         f"{rows[-1]['time']:.3g} | {observed_order(rows):.2f} | "
                         + " | ".join("–" if c is None else f"{c:.3g} s" for c in cells) + " |")
        reference = record["reference"]
        if reference != "exact":
            lines.append(f"\nReference: fronttrack at δ = {reference['delta']:.3g} "
                         f"({reference['events']} events, {reference['time']:.0f} s).")
        if "fronttrack" in methods and len(methods) > 1:
            lines.append("")
            for target in TARGETS:
                mine = time_to(methods["fronttrack"], target)
                theirs = {m: time_to(rows, target) for m, rows in methods.items()
                          if m != "fronttrack"}
                theirs = {m: t for m, t in theirs.items() if t is not None}
                if mine is None:
                    continue
                if not theirs:
                    lines.append(f"- L1 {target:.0e}: fronttrack {mine:.3g} s; "
                                 "no Clawpack run reached it")
                    continue
                best = min(theirs, key=theirs.get)
                lines.append(f"- L1 {target:.0e}: fronttrack {mine:.3g} s, best Clawpack "
                             f"({LABELS[best]}) {theirs[best]:.3g} s: "
                             f"**{theirs[best] / mine:.2g}×**")
    return "\n".join(lines)


STYLE = {"fronttrack": ("#2a78d6", "o"), "classic_mc": ("#eb6834", "s"),
         "sharpclaw_weno5": ("#1baf7a", "D"), "classic_godunov": ("#8a8984", "^")}


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records = list(results["problems"].values())
    cols = 2 if len(records) > 1 else 1
    rows_ = math.ceil(len(records) / cols)
    fig, axes = plt.subplots(rows_, cols, figsize=(6.2 * cols, 4.4 * rows_),
                             constrained_layout=True, squeeze=False)
    for ax, record in zip(axes.flat, records):
        for method, rows in record["methods"].items():
            color, marker = STYLE[method]
            ax.plot([r["err"] for r in rows], [max(r["time"], 1e-4) for r in rows],
                    color=color, marker=marker, ms=6, lw=2, label=LABELS[method], zorder=3)
        ax.set(xscale="log", yscale="log", title=record["title"],
               xlabel="L¹ error at the final time  (→ more accurate)",
               ylabel="wall time, s")
        ax.invert_xaxis()
        ax.title.set(fontsize=11, ha="left", x=0.0)
        ax.grid(True, color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.flat[len(records):]:
        ax.remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(path, dpi=130)
    print("wrote", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", nargs="+", choices=list(PROBLEMS))
    parser.add_argument("--budget", type=float, default=60.0,
                        help="stop refining a method after a run longer than this (s)")
    parser.add_argument("--max-level", type=int, default=17,
                        help="finest fronttrack delta is 2^-max_level")
    parser.add_argument("--reference-level", type=int, default=19,
                        help="delta = 2^-level for the cubic_sine reference")
    parser.add_argument("--skip-clawpack", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "benchmarks" / "clawpack.json")
    parser.add_argument("--summarize", type=Path, metavar="JSON",
                        help="only print the table and redraw the plot for saved results")
    args = parser.parse_args()
    if args.summarize:
        results, png = json.loads(args.summarize.read_text()), args.summarize.with_suffix(".png")
    else:
        results, png = measure(args), args.out.with_suffix(".png")
    print(summary(results))
    plot(results, png)


if __name__ == "__main__":
    main()
