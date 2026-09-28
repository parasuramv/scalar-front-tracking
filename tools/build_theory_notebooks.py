"""Build the illustrative ("theory") notebooks from readable cell lists.

Unlike the two tool notebooks, these do NOT bundle the solver source. The setup
cell looks for ``fronttrack`` in this order:

1. the checkout (``./src`` or ``../src``), or an already installed package;
2. ``pip install git+https://github.com/REPO`` (works once the repo is public);
3. in Colab only, the same install authenticated with a Colab secret named
   ``GITHUB_TOKEN`` (a fine-grained, read-only token for this repository).

If all three fail it stops with an explanation instead of a pip traceback, and
distinguishes a missing secret from one the notebook was not allowed to read.
Execute and save outputs with:

    python tools/build_theory_notebooks.py --execute
"""
from pathlib import Path
import argparse

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
REPO = "parasuramv/scalar-front-tracking"

SETUP = f'''\
# Setup: use the local checkout if present, otherwise install from GitHub.
# Private repo + Colab: add a secret GITHUB_TOKEN (key icon, left sidebar) holding
# a fine-grained token with read access to {REPO}. See notebooks/README.md.
import os, sys, subprocess, importlib, importlib.util
from pathlib import Path
REPO = "{REPO}"

def _pip_install(target):
    # No terminal prompt: a private repo must fail fast, not wait for a password.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", target], check=True,
                   capture_output=True, env={{**os.environ, "GIT_TERMINAL_PROMPT": "0"}})

for candidate in (Path.cwd() / "src", Path.cwd().parent / "src"):
    if (candidate / "fronttrack").is_dir():
        sys.path.insert(0, str(candidate))
        break
if importlib.util.find_spec("fronttrack") is None:
    try:
        _pip_install(f"git+https://github.com/{{REPO}}.git")
    except subprocess.CalledProcessError:
        token = None
        hint = "add a GITHUB_TOKEN secret with read access to the repository"
        if "google.colab" in sys.modules:
            from google.colab import userdata
            try:
                token = userdata.get("GITHUB_TOKEN")
            except Exception as err:
                # Two different failures: no such secret, or the secret exists but
                # this notebook's access toggle is off. Tell the user which.
                if type(err).__name__ == "NotebookAccessError":
                    hint = ("switch on notebook access for the GITHUB_TOKEN secret "
                            "(key icon in the left sidebar)")
        if not token:
            raise RuntimeError(
                f"Could not install fronttrack from github.com/{{REPO}} (the repository "
                f"may still be private). Run this notebook from a checkout, or in Colab "
                f"{{hint}}.") from None
        try:
            _pip_install(f"git+https://x-access-token:{{token}}@github.com/{{REPO}}.git")
        except subprocess.CalledProcessError:
            # 'from None' keeps the failed command, which contains the token, out of view.
            raise RuntimeError("GITHUB_TOKEN was found but the install failed: check "
                               f"that the token can read {{REPO}}.") from None
    importlib.invalidate_caches()
for name in ("matplotlib", "ipywidgets"):
    if importlib.util.find_spec(name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", name])
if "google.colab" in sys.modules:
    from google.colab import output
    output.enable_custom_widget_manager()
%matplotlib inline'''

STYLE = '''\
import numpy as np
import matplotlib.pyplot as plt
import fronttrack
from fronttrack import DiscreteFlux, FrontTracker, solve_riemann

# One small, fixed palette for the whole series.
C_FLUX, C_POLY, C_ENV, C_EXACT = "#9a9a94", "#2a78d6", "#eb6834", "#222222"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "lines.linewidth": 2})
print("fronttrack", fronttrack.__version__)'''


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def riemann_notebook():
    cells = [
        md(r"""
# 03 · The Riemann problem, one envelope at a time

Front tracking is a single idea applied over and over: **solve a Riemann problem
exactly, move the resulting fronts, and when two fronts meet, solve a new Riemann
problem.** This notebook is about that one building block.

We look at
$$
u_t + f(u)_x = 0,\qquad u(x,0)=\begin{cases}u_L,& x<0,\\ u_R,& x>0,\end{cases}
$$
and ask three questions:

1. What is the entropy solution for a general (possibly nonconvex) flux $f$?
2. What does it become when $f$ is replaced by a **polygonal** flux $f_\delta$?
3. How far is the polygonal answer from the true one?

*Prerequisites: the Rankine–Hugoniot condition and the idea of an entropy
solution. Reference: Holden & Risebro, *Front Tracking for Hyperbolic
Conservation Laws*, Ch. 2.*
"""),
        code(SETUP),
        code(STYLE),
        md(r"""
## 1. The envelope rule

For a scalar law the entropy solution of the Riemann problem is self-similar,
$u(x,t)=w(x/t)$, and it is read off from a single convex-analysis object
(Oleinik's rule, H&R §2.2):

* if $u_L<u_R$, take the **lower convex envelope** $\check f$ of $f$ on $[u_L,u_R]$;
* if $u_L>u_R$, take the **upper concave envelope** $\hat f$ of $f$ on $[u_R,u_L]$.

Where the envelope *coincides* with $f$ the solution is a rarefaction, with
$w(\xi) = (\check f')^{-1}(\xi)$. Where the envelope is a *chord* it jumps across
that chord in a shock, and the chord's slope is the Rankine–Hugoniot speed. A chord that is tangent to $f$ at one end
gives a shock glued to a rarefaction: this is a **contact** or compound wave.

For convex $f$ this recovers the familiar dichotomy: an increasing jump makes a
rarefaction ($\check f=f$), and a decreasing jump makes a shock ($\hat f$ is the single chord).
"""),
        md(r"""
## 2. Polygonal flux: rarefactions become staircases

Front tracking replaces $f$ by its piecewise-linear interpolant $f_\delta$ on a
state grid $u_0<u_1<\dots<u_N$ with spacing $\delta$. The envelope of a
polygon is again a polygon, and its vertices are grid nodes. Therefore:

> **Every Riemann solution with flux $f_\delta$ is a finite fan of discontinuities**
> between grid states. Each edge of the envelope is one front, and its speed is the slope of that edge.

A rarefaction is therefore a staircase of small jumps of height $\delta$. The fan has
finitely many fronts, which is why the whole method needs no mesh and no CFL
condition. Here is Burgers' flux with 9 nodes, the increasing jump $-1\to 1$:
"""),
        code(r'''
def fan_figure(f, uL, uR, nodes, t=1.0, exact=None, title=""):
    """Left: f, f_delta and the envelope. Right: the fan at time t vs. exact."""
    lo, hi = min(uL, uR), max(uL, uR)
    flux = DiscreteFlux(f, np.linspace(-1, 1, nodes))
    iL, iR = flux.state_index(uL), flux.state_index(uR)
    waves = solve_riemann(iL, iR, flux)
    hull = [waves[0][0]] + [b for _, b, _ in waves] if waves else [iL]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    u = np.linspace(-1, 1, 801)
    a.plot(u, f(u), color=C_FLUX, lw=1.5, label="$f$")
    a.plot(flux.u_grid, flux.f_values, "o-", color=C_POLY, ms=4, lw=1.2,
           label=r"$f_\delta$")
    a.plot(flux.u_grid[hull], flux.f_values[hull], "-", color=C_ENV, lw=2.5,
           label="envelope")
    a.axvspan(lo, hi, color=C_ENV, alpha=0.06)
    a.set(xlabel="$u$", ylabel="flux", title="Flux and envelope")
    a.legend(frameon=False)

    # The profile at time t: constant states separated by fronts at x = s t.
    xs = np.array([s * t for *_, s in waves])
    states = [flux.u_grid[iL]] + [flux.u_grid[r] for _, r, _ in waves]
    span = max(1.0, *(abs(xs) if len(xs) else [0])) * 1.3
    edges = np.r_[-span, xs, span]
    for k, s in enumerate(states):
        b.hlines(s, edges[k], edges[k + 1], color=C_POLY, lw=2)
    b.vlines(xs, [flux.u_grid[l] for l, _, _ in waves],
             [flux.u_grid[r] for _, r, _ in waves], color=C_POLY, lw=1)
    if exact is not None:
        x = np.linspace(-span, span, 1201)
        b.plot(x, exact(x, t), "--", color=C_EXACT, lw=1.5, label="exact, flux $f$")
        b.legend(frameon=False, loc="best")
    b.set(xlabel="$x$", ylabel="$u$", title=f"Solution at t = {t:g}  ({len(waves)} fronts)")
    fig.suptitle(title)
    plt.show()
    return waves

burgers = lambda u: 0.5 * u**2
rare_exact = lambda x, t: np.clip(x / t, -1, 1)
waves = fan_figure(burgers, -1, 1, 9, exact=rare_exact,
                   title=r"Burgers, $u_L=-1<u_R=1$: rarefaction → staircase")
for l, r, s in waves:
    print(f"front  node {l}->{r}   speed {s:+.3f}")
'''),
        md(r"""
Each front joins adjacent grid states, and its speed is the **average** of
$f'$ over that cell, $(f(u_{i+1})-f(u_i))/\delta$. For Burgers this is the
midpoint $u_i+\delta/2$. The staircase meets the exact fan $u=x/t$ at the
middle of each step.

The decreasing jump $1\to-1$ is the opposite case: the concave envelope is one
chord, so there is a single shock. Its speed is $(f(1)-f(-1))/2=0$. There is
**no discretisation error at all** here, since the shock is exact for any $\delta$ that contains both states as nodes.
"""),
        code(r'''
shock_exact = lambda x, t: np.where(x < 0, 1.0, -1.0)
fan_figure(burgers, 1, -1, 9, exact=shock_exact,
           title=r"Burgers, $u_L=1>u_R=-1$: one chord, one shock");
'''),
        md(r"""
## 3. A nonconvex flux: the compound wave

Take $f(u)=u^3$ with $u_L=-1<u_R=1$. The lower convex envelope on $[-1,1]$ follows the chord from
$(-1,-1)$ until the chord becomes tangent to $f$, and then follows $f$ itself. Tangency at $u^*$ means
$$
f'(u^*) = \frac{f(u^*)-f(-1)}{u^*+1}
\iff 3u^{*2}(u^*+1) = u^{*3}+1
\iff 2u^{*2}+u^*-1 = 0
\iff u^* = \tfrac12 .
$$
The exact solution is therefore a shock from $-1$ to $\tfrac12$ travelling at speed
$f'(\tfrac12)=\tfrac34$, followed without any gap by a rarefaction $u=\sqrt{x/(3t)}$ for
$\tfrac34\le x/t\le 3$. This shock is **sonic**: it moves at the characteristic speed of its right state
(Oleinik's condition holds with equality).
"""),
        code(r'''
cubic = lambda u: u**3
def compound_exact(x, t):
    xi = x / t
    return np.where(xi < 0.75, -1.0, np.where(xi > 3, 1.0, np.sqrt(np.clip(xi, 0, None) / 3)))

waves = fan_figure(cubic, -1, 1, 17, exact=compound_exact,
                   title=r"$f=u^3$, $u_L=-1<u_R=1$: shock + attached rarefaction")
l, r, s = waves[0]
print(f"first front: -1 -> {DiscreteFlux(cubic, np.linspace(-1,1,17)).u_grid[r]:+.3f}, speed {s:.3f}"
      "   (exact: -1 -> 0.5, speed 0.75)")
'''),
        md(r"""
With 17 nodes, $\tfrac12$ is a grid node, so the first front is *exactly* the true shock.
Try `nodes=16` in the cell above. Then $u^*$ is no longer a node, and the discrete
envelope chooses a nearby node. The shock speed error is $O(\delta)$.
"""),
        md(r"""
## 4. How the envelope is actually computed

`solve_riemann` builds the envelope with a single monotone-stack pass (Andrew's
monotone chain, a close relative of Graham's scan). It visits the nodes
between $u_L$ and $u_R$ in order. Each new node pops every earlier node that it
exposes as lying *above* the lower hull, which is detected by a slope that fails to increase. Every node is
pushed once and popped at most once, so the cost is $O(|i_R-i_L|)$.

The cell below re-implements the loop with a record of each step, checks the result against
the library, and draws four stages for the cubic.
"""),
        code(r'''
def hull_trace(flux, iL, iR):
    """Lower (convex) hull for iL<iR, recording the stack after every node."""
    s, v = flux.u_grid, flux.f_values
    hull, stages = [], []
    for n in range(iL, iR + 1):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (v[b] - v[a]) / (s[b] - s[a]) >= (v[n] - v[a]) / (s[n] - s[a]):
                hull.pop()           # b lies on or above the chord a -> n
            else:
                break
        hull.append(n)
        stages.append(list(hull))
    return stages

flux = DiscreteFlux(cubic, np.linspace(-1, 1, 11))
stages = hull_trace(flux, 0, 10)
library = [0] + [r for _, r, _ in solve_riemann(0, 10, flux)]
assert stages[-1] == library, (stages[-1], library)

fig, axes = plt.subplots(1, 4, figsize=(13, 3), sharey=True, constrained_layout=True)
for ax, k in zip(axes, (3, 6, 8, 10)):
    ax.plot(flux.u_grid, flux.f_values, "o-", color=C_POLY, ms=4, lw=1)
    ax.plot(flux.u_grid[k], flux.f_values[k], "o", color=C_EXACT, ms=8, mfc="none")
    h = stages[k]
    ax.plot(flux.u_grid[h], flux.f_values[h], "-", color=C_ENV, lw=2.5)
    ax.set(title=f"after node {k}", xlabel="$u$")
axes[0].set_ylabel(r"$f_\delta$")
plt.show()
print("hull nodes:", library)
'''),
        md(r"""
Up to node 7 ($u=0.4$) each new node pops everything in between, so the stack is always one
chord from $u_0=-1$. From there on, $f_\delta$ curves upward faster than any chord from $-1$
(the continuous tangency is at $u^*=\tfrac12$, which lies between nodes 7 and 8), so every later node survives.
The hull ends as the chord $0\to7$ followed by $f_\delta$ itself.
"""),
        md(r"""
## 5. How good is the polygonal answer?

We measure $\|u_\delta(\cdot,1)-u(\cdot,1)\|_{L^1}$ against the exact solutions above as
$\delta\to0$. For Riemann data, standard theory (H&R Thm 2.4 and §2.3) gives an $O(\delta)$ bound.
The rarefaction's error can be computed by hand: each of the $2/\delta$ steps
misses a triangle pair of area $\delta^2/4$ at $t=1$. The total is $\delta/2$.
"""),
        code(r'''
def l1_error(f, uL, uR, exact, nodes, t=1.0, window=(-2.0, 4.0), n=400_001):
    flux = DiscreteFlux(f, np.linspace(-1, 1, nodes))
    tr = FrontTracker.from_values(flux, [0.0], [uL, uR], record_history=False).run(t)
    x = np.linspace(*window, n)
    return np.trapezoid(np.abs(tr.sample(x) - exact(x, t)), x)

nodes = np.array([9, 17, 33, 65, 129, 257])
delta = 2 / (nodes - 1)
cases = {"Burgers rarefaction": (burgers, -1, 1, rare_exact),
         "u³ compound wave": (cubic, -1, 1, compound_exact),
         "Burgers shock": (burgers, 1, -1, shock_exact)}
errors = {k: np.array([l1_error(f, a, b, ex, n) for n in nodes])
          for k, (f, a, b, ex) in cases.items()}

fig, ax = plt.subplots(figsize=(6.5, 4))
for name, c in (("Burgers rarefaction", C_POLY), ("u³ compound wave", C_ENV)):
    ax.loglog(delta, errors[name], "o-", color=c, ms=5, label=name)
ax.loglog(delta, delta / 2, ":", color=C_EXACT, lw=1.2, label=r"$\delta/2$")
ax.set(xlabel=r"state spacing $\delta$", ylabel=r"$L^1$ error at $t=1$",
       title="Riemann problems: first-order in δ")
ax.legend(frameon=False)
plt.show()

print(f"{'delta':>8} " + " ".join(f"{k:>22}" for k in errors))
for i, d in enumerate(delta):
    print(f"{d:8.4f} " + " ".join(f"{errors[k][i]:22.3e}" for k in errors))
'''),
        md(r"""
The rarefaction lands on $\delta/2$ exactly. The compound wave is also first order.
The shock is not plotted because its error is exactly zero: a shock between grid nodes is exact.
In the polygonal world all of the error sits in the rarefactions, which is the conceptual reason the full
method converges at rate $O(\delta)$ in $L^1$ (up to the initial-data error).
"""),
        md(r"""
## 6. Try it yourself

Change the flux, the two states or the number of nodes. The sliders only work in Jupyter or Colab,
since GitHub shows the saved picture. Good things to try:

* `sin(3u)` with a long jump gives several alternating shocks and rarefactions.
* `u(1-u)` (traffic flow) is concave, so an *increasing* jump gives the shock.
* Burgers $1\to-1$ with an **odd** number of nodes, then with an even number.
"""),
        code(r'''
import ipywidgets as widgets
fluxes = {"Burgers u²/2": burgers, "cubic u³": cubic, "traffic u(1-u)": lambda u: u * (1 - u),
          "sin(3u)": lambda u: np.sin(3 * u), "Buckley–Leverett":
          lambda u: (u + 1)**2 / ((u + 1)**2 + 0.5 * (1 - u)**2)}
controls = dict(
    name=widgets.Dropdown(options=list(fluxes), value="sin(3u)", description="flux"),
    uL=widgets.FloatSlider(value=-1, min=-1, max=1, step=0.05, description="u_L"),
    uR=widgets.FloatSlider(value=1, min=-1, max=1, step=0.05, description="u_R"),
    nodes=widgets.IntSlider(value=41, min=3, max=201, description="nodes"))
def show(name, uL, uR, nodes):
    if abs(uL - uR) < 1e-12:
        print("Equal states: no wave.")
        return
    fan_figure(fluxes[name], uL, uR, nodes, title=name)
display(widgets.HBox(list(controls.values()), layout=widgets.Layout(flex_flow="row wrap")),
        widgets.interactive_output(show, controls))
'''),
        md(r"""
---
**Next:** *04 · Interactions*. It covers what happens when two of these fans collide, and why
only neighbouring fronts ever need to be checked.
"""),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "03_riemann_problem.ipynb", "toc_visible": True},
    })
    return nb


NOTEBOOKS = {"03_riemann_problem.ipynb": riemann_notebook}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="run and save outputs")
    args = parser.parse_args()
    for name, build in NOTEBOOKS.items():
        nb = build()
        path = ROOT / "notebooks" / name
        if args.execute:
            from nbclient import NotebookClient
            NotebookClient(nb, timeout=600, kernel_name="python3",
                           resources={"metadata": {"path": str(path.parent)}}).execute()
        nbf.write(nb, path)
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
