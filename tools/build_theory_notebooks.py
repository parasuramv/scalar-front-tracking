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
between $u_L$ and $u_R$ in order. With $a,b$ the top two stack entries and $n$ the new
node, $b$ survives only if it lies *strictly below* the chord $a\to n$; otherwise it is popped
and the test repeats. Every node is pushed once and popped at most once, so the cost is
$O(|i_R-i_L|)$.

"Strictly below" needs care in floating point, and the cubic shows why: see below the figure.

The cell below re-implements the loop with a record of each step, checks the result against
the library, and draws four stages for the cubic.
"""),
        code(r'''
def hull_trace(flux, iL, iR):
    """Lower (convex) hull for iL<iR, recording the stack after every node."""
    s, v = flux.u_grid, flux.f_values
    u_scale = max(abs(s[0]), abs(s[-1]))
    hull, stages = [], []
    for n in range(iL, iR + 1):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            rise, run = v[n] - v[a], s[n] - s[a]
            gap = (v[b] - v[a]) - rise * ((s[b] - s[a]) / run)   # height of b above chord a -> n
            tol = 16 * np.finfo(float).eps * (max(abs(v[a]), abs(v[b]), abs(v[n]))
                                              + abs(rise / run) * u_scale)
            if gap >= -tol:
                hull.pop()           # b lies on or above the chord a -> n (up to roundoff)
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
Up to node 8 ($u=0.6$) each new node pops everything in between, so the stack is always one
chord from $u_0=-1$. From node 9 on, $f_\delta$ curves upward faster than any chord from $-1$,
so every later node survives. The hull ends as the chord $0\to8$ followed by $f_\delta$ itself.

Node 8, not node 7, is the end of the chord. For $f=u^3$ the chord slope from $u_a$ to $u_b$ is
$u_a^2+u_au_b+u_b^2$, which for $u_a=-1$ is symmetric in $u_b$ about $\tfrac12$, the continuous
tangency point $u^*$. Nodes 7 and 8 ($u=0.4$ and $0.6$) straddle $u^*$ symmetrically, so the nodes
$0,7,8$ are exactly collinear and the envelope is the single chord $0\to8$, one shock of speed $0.76$.

In floating point the two slopes come out as `0.76` and `0.7600000000000003`. An exact
comparison keeps node 7 and splits the shock into two fronts moving together, a zero-width
state that is not in the entropy solution. The loop above therefore compares the height of
$b$ above the chord with a roundoff allowance: $16\varepsilon$ times the size of the numbers
that enter the test. Uniform grids for $u^3$ contain many such collinear triples, one for every
pair of nodes placed symmetrically about a tangency point.
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


def interactions_notebook():
    cells = [
        md(r"""
# 04 · Interactions: the event loop

Notebook 03 solved one Riemann problem. General step data has many jumps, and
each one opens its own fan of fronts. Between collisions every front moves on a
straight line. When two fronts meet, the solution near the meeting point is
again two constant states separated by a single jump, so the next step is
**another Riemann problem**. Front tracking is exactly this loop:

> find the next collision → move every front up to that time → replace the colliding
> fronts with the fan of the new Riemann problem → repeat.

The loop raises four questions:

1. Which pairs of fronts do we have to check? (§1–2)
2. What comes out of a collision? (§3 shock–shock, §4 shock–rarefaction)
3. What if more than two fronts reach the same point at the same time? (§5)
4. Does the loop stop? How many fronts and events are there? (§6)

*Reference: Holden & Risebro, *Front Tracking for Hyperbolic Conservation Laws*, Ch. 2.*
"""),
        code(SETUP),
        code(STYLE),
        code(r'''
from matplotlib.collections import LineCollection
from fronttrack import initial_fronts, next_interaction, resolve_interaction

burgers = lambda u: 0.5 * u**2
sin3 = lambda u: np.sin(3 * u)

def xt_plot(ax, segments, events=(), title=""):
    """Draw front paths (x0, t0, x1, t1) in the x-t plane; mark events as dots."""
    ax.add_collection(LineCollection([((a, b), (c, d)) for a, b, c, d in segments],
                                     colors=C_POLY, linewidths=1))
    if len(events):
        ax.plot([e[1] for e in events], [e[0] for e in events], "o",
                color=C_ENV, ms=3.5, zorder=3)
    ax.autoscale()
    ax.set(xlabel="$x$", ylabel="$t$", title=title)
'''),
        md(r"""
## 1. Only neighbours can collide first

Order the fronts by position, $x_1(t)\le x_2(t)\le\dots\le x_N(t)$. Until the first
collision each $x_j$ is affine in $t$, and the order cannot change, because two
fronts that swap places must pass through a common point, and that would be a collision.

**Claim.** The first collision time is the minimum over *adjacent* pairs only.

*Proof.* Suppose fronts $i<k$ meet first, at time $t^*$, with $x_i(t^*)=x_k(t^*)$.
Take any $j$ with $i<j<k$. On $[0,t^*)$ we have $x_i\le x_j\le x_k$, and by continuity
this still holds at $t^*$. So $x_j(t^*)$ is squeezed to the same point, and the
adjacent pair $(i,i+1)$ has also met by $t^*$. $\square$

So the next event is found by scanning $N-1$ adjacent pairs instead of all
$\binom N2$ pairs. A pair $(j,j+1)$ collides only if it is closing,
$s_j>s_{j+1}$, and then after time $(x_{j+1}-x_j)/(s_j-s_{j+1})$. This is
`next_interaction`. The same argument gives a second saving: an event
changes only the fronts at one point, so only the (at most two) pairs next to it get
new collision times. A priority queue would therefore cost $O(\log N)$ per event instead of
$O(N)$. `fronttrack` keeps the readable $O(N)$ scan.
"""),
        md(r"""
## 2. The event loop in a dozen lines

Here is the whole algorithm written out with the library's two building blocks.
It records every event (time, place, fronts in, fronts out) and every path segment.
"""),
        code(r'''
def event_loop(flux, positions, states, T):
    """Naive front tracking: resolve one colliding pair at a time."""
    fronts = initial_fronts(positions, states, flux).fronts
    t, events, segments = 0.0, [], []

    def advance(dt):
        for f in fronts:
            segments.append((f.x, t, f.x + f.speed * dt, t + dt))
            f.x += f.speed * dt

    while True:
        dt, i = next_interaction(fronts)        # scan adjacent pairs only
        if i is None or t + dt > T:             # nothing more before T
            advance(T - t)
            return fronts, events, segments
        advance(dt)
        t += dt
        new = resolve_interaction(fronts[i], fronts[i + 1], flux)  # Riemann (u_left, u_right)
        events.append((t, fronts[i].x, 2, len(new)))
        fronts[i:i + 2] = new

rng = np.random.default_rng(1)
flux = DiscreteFlux(burgers, np.linspace(-1, 1, 21))
positions = np.sort(rng.uniform(-3, 3, 12))
states = flux.state_index(rng.uniform(-1, 1, 13))

fronts, events, segments = event_loop(flux, positions, states, T=6)
library = FrontTracker(flux, positions, states).run(6)
assert [(f.iL, f.iR) for f in fronts] == [(f.iL, f.iR) for f in library.fronts]
assert np.allclose([f.x for f in fronts], [f.x for f in library.fronts])

fig, ax = plt.subplots(figsize=(10, 4.2))
xt_plot(ax, segments, events, title=f"Burgers, 12 random jumps: {len(events)} events")
ax.set_xlim(-4, 5)
plt.show()
print(f"fronts at t=0: {len(initial_fronts(positions, states, flux).fronts)},"
      f" at t=6: {len(fronts)}, events: {len(events)}   (matches FrontTracker)")
'''),
        md(r"""
Each fan in the picture is a rarefaction staircase from notebook 03, and each dot is
one Riemann problem solved at a collision. The steps of a staircase are parallel
and never meet each other. They end only when a shock catches them. The shocks
absorb everything they touch, which is the pattern of the next two sections.
"""),
        md(r"""
## 3. Shock meets shock

Burgers with $u = 1, 0, -1$ and jumps at $x=\mp1$. Both jumps are decreasing, so each is a single
shock. The speeds are $\tfrac12(1+0)=\tfrac12$ and $\tfrac12(0-1)=-\tfrac12$, so they meet at
$(x,t)=(0,2)$. The collision leaves the Riemann problem $1\to-1$, which is one shock of speed $0$.
"""),
        code(r'''
flux = DiscreteFlux(burgers, np.linspace(-1, 1, 3))
fronts, events, segments = event_loop(flux, [-1.0, 1.0], flux.state_index([1, 0, -1]), T=4)
fig, ax = plt.subplots(figsize=(6, 3.6))
xt_plot(ax, segments, events, title="Two shocks merge into one")
plt.show()
for t, x, n_in, n_out in events:
    print(f"event at t={t:g}, x={x:g}: {n_in} fronts in, {n_out} out")
for f in fronts:
    print(f"outgoing: {flux.u_grid[f.iL]:g} → {flux.u_grid[f.iR]:g}, speed {f.speed:g}")
'''),
        md(r"""
Every quantity here is a Rankine–Hugoniot speed between grid states, so this picture is
the **exact** entropy solution for any $\delta$ that has $1,0,-1$ as nodes. Shocks
cost nothing in front tracking. All of the approximation lives in rarefactions,
as in notebook 03.
"""),
        md(r"""
## 4. Shock meets rarefaction

Take Burgers with $u_0 = 0,\,1,\,0$ and jumps at $x=0$ and $x=1$. The jump at $0$ opens a
rarefaction $u=x/t$, and the jump at $1$ is a shock of speed $\tfrac12$. The fan's leading edge
(speed $1$) catches the shock at $t=2$. After that, the shock sits at the right end of the
fan, and conservation of mass fixes where that is:
$$
\int_0^{X(t)} \frac{x}{t}\,dx = 1 \;\Longrightarrow\; X(t)=\sqrt{2t}\qquad(t\ge2),
\qquad X(t) = 1+\tfrac t2\quad(t\le 2).
$$
With a polygonal flux the fan is a staircase of $N=1/\delta$ steps. The shock absorbs them
**one at a time**. Each absorption is one event, and afterwards the shock is slower because
its left state has dropped by $\delta$. The curved shock path becomes a polygon.
"""),
        code(r'''
def shock_path_error(N, times):
    flux = DiscreteFlux(burgers, np.linspace(0, 1, N + 1))
    tr = FrontTracker.from_values(flux, [0.0, 1.0], [0, 1, 0], record_history=False)
    err = []
    for t in times:
        tr.run(t)
        exact = 1 + t / 2 if t <= 2 else np.sqrt(2 * t)
        err.append(abs(tr.fronts[-1].x - exact))   # the shock is the rightmost front
    return max(err)

N = 8
flux = DiscreteFlux(burgers, np.linspace(0, 1, N + 1))
fronts, events, segments = event_loop(flux, [0.0, 1.0], flux.state_index([0, 1, 0]), T=8)

Ns = np.array([8, 16, 32, 64, 128, 256])
times = np.linspace(0.05, 8, 160)
errors = np.array([shock_path_error(n, times) for n in Ns])

fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4), constrained_layout=True)
xt_plot(a, segments, events, title=f"δ = 1/{N}: the shock eats the staircase")
t = np.linspace(0, 8, 400)
a.plot(np.where(t <= 2, 1 + t / 2, np.sqrt(2 * t)), t, "--", color=C_EXACT, lw=1.3,
       label="exact shock")
a.set_xlim(-0.2, 4.6)
a.legend(frameon=False, loc="upper left")
b.loglog(1 / Ns, errors, "o-", color=C_POLY, ms=5, label=r"max shock-position error, $t\leq 8$")
b.loglog(1 / Ns, errors[0] * (Ns[0] / Ns) ** 2, ":", color=C_EXACT, label=r"slope 2")
b.set(xlabel=r"$\delta$", title="The shock position is second order")
b.legend(frameon=False)
plt.show()
rates = np.log2(errors[:-1] / errors[1:])
print("observed rates:", np.round(rates, 2))
'''),
        md(r"""
This is sharper than the $O(\delta)$ $L^1$ error of notebook 03, and conservation explains
why. The shock sits where the mass to its left equals $1$, both for the exact solution and
for the staircase. The staircase differs from $x/t$ by $\pm\delta/2$, but on each **full**
step the signed difference integrates to zero: the step meets $x/t$ at its midpoint, and the
two triangles cancel. Only the half-step at $x=0$ and the partial step cut off by the shock
contribute. Each of these is $O(\delta^2 t)$ in mass, so $|X_\delta - X| = O(\delta^2)$ on
bounded time intervals.

The $L^1$ error adds up $|u_\delta-u|$, where nothing cancels, so it stays first order.
The shock, a conserved-mass feature, is located to second order.
"""),
        md(r"""
## 5. Clusters: more than two fronts at one point

The proof in §1 shows something else too: when $i$ and $k$ meet, every front between them
is at the same point. So collisions can involve **three or more fronts at once**, and on a
uniform grid with rational speeds this is common, not exceptional. The correct local problem
is then the Riemann problem between the two *outermost* states, because the intermediate
states have zero width at that instant.

`FrontTracker` does this: it gathers all fronts within `position_tol` (plus roundoff) of the
collision point into a *cluster* and resolves it in one step. The naive loop above does not.
It resolves one pair and leaves the remaining fronts to be found by the next scan
(with waiting time $0$). For Burgers this usually gives the same answer:
"""),
        code(r'''
flux = DiscreteFlux(burgers, np.linspace(0, 3, 4))
speeds = np.array([2.5, 1.5, 0.5])                 # shocks 3→2, 2→1, 1→0
positions, states = -speeds, [3, 2, 1, 0]          # all three reach x=0 at t=1
naive, ev, _ = event_loop(flux, positions, states, T=2)
tracker = FrontTracker(flux, positions, states).run(2)
print("naive loop:  ", len(ev), "events ->", [(f.iL, f.iR) for f in naive])
print("FrontTracker:", tracker.event_count, "event  ->", [(f.iL, f.iR) for f in tracker.fronts])
'''),
        md(r"""
Same answer, but twice the events. With a nonconvex flux the naive loop also leaves
**debris**. Below, three fronts of $f(u)=\sin 3u$ meet at $(0,1)$. The outer states are equal,
so the local Riemann problem is trivial and nothing should come out. The naive loop resolves
the first pair and is left with two fronts, $-\tfrac56\to-1$ and $-1\to-\tfrac56$. These are the same chord
traversed in both directions, so they have the *same* speed. They never close, never collide,
and ride together forever around a state of zero width.
"""),
        code(r'''
flux = DiscreteFlux(sin3, np.linspace(-1, 1, 13))
states = [1, 2, 0, 1]                               # u = -5/6, -2/3, -1, -5/6
speeds = np.array([solve_riemann(a, b, flux)[0][2] for a, b in zip(states, states[1:])])
positions = -speeds                                # all three reach x=0 at t=1
naive, ev, seg = event_loop(flux, positions, states, T=3)
tracker = FrontTracker(flux, positions, states).run(3)

fig, ax = plt.subplots(figsize=(6, 3.6))
xt_plot(ax, seg, ev, title="naive loop: two fronts leave, riding together")
plt.show()
x = np.linspace(-8, 8, 4001)
naive_u = np.full_like(x, flux.u_grid[states[0]])
for f in naive:
    naive_u[x >= f.x] = flux.u_grid[f.iR]
print("naive loop:  ", ", ".join(f"{flux.u_grid[f.iL]:.3f}→{flux.u_grid[f.iR]:.3f} "
                                  f"(speed {f.speed:.4f})" for f in naive))
print("FrontTracker:", len(tracker.fronts), "fronts")
print("L1 difference of the two solutions at t=3:",
      np.trapezoid(np.abs(naive_u - tracker.sample(x)), x))
'''),
        md(r"""
So the naive loop gets the **solution** right but the **representation** wrong. In every
triple collision we tried, for several fluxes, the sampled solutions agree. The debris,
however, is scanned at every later event, is counted as fronts, and re-enters every Riemann
problem it later takes part in. Resolving the whole cluster against its two outer states
gives the minimal fan directly.

Exact multiple collisions have measure zero for *random* data, which is why the naive loop
matched the library in §2. Polygonal fluxes on uniform grids produce rational speeds, though,
and structured data hits these coincidences often. Floating-point positions rarely meet
*exactly*, so the cluster test uses a tolerance (`position_tol`, default $10^{-12}$, plus a
relative roundoff allowance).
"""),
        md(r"""
## 6. How many fronts? How many events?

For the loop to be an algorithm it has to stop: finitely many events on $[0,T]$.

**Lemma (strictly convex $f$).** A collision of two fronts produces at most one front.

*Proof.* For strictly convex $f_\delta$, an increasing jump is a single front only if it joins
adjacent nodes $(a, a{+}1)$, and a decreasing jump is always a single shock. Two steps
$(a,a{+}1),(a{+}1,a{+}2)$ have increasing speeds and never meet, so at least one incoming
front is a shock. Checking the two remaining cases (step then shock, shock then step) shows
that the outer states satisfy $u_{\rm right}\le u_{\rm left}$. So the outgoing Riemann problem
is a single shock or nothing. $\square$

So every binary collision removes at least one front, and
$E(t)\le N(0)-N(t)\le N(0)$: the number of events is bounded by the initial number of fronts.

For a **nonconvex** flux the lemma fails. A collision can send out as many fronts as came in:
"""),
        code(r'''
flux = DiscreteFlux(sin3, np.linspace(-1, 1, 21))
a, b, c = flux.state_index([-0.3, 0.6, 0.5])
show = lambda waves: ", ".join(f"{flux.u_grid[l]:+.1f}→{flux.u_grid[r]:+.1f} (speed {s:.3f})"
                                for l, r, s in waves)
print("in: ", show(solve_riemann(a, b, flux) + solve_riemann(b, c, flux)))
print("out:", show(solve_riemann(a, c, flux)))
'''),
        md(r"""
With $f(u)=\sin 3u$ on 21 nodes, the shock $-0.3\to0.6$ catches the step $0.6\to0.5$.
The outgoing problem $-0.3\to0.5$ spans the inflection point at $u=0$, so its envelope has two
edges: two fronts go in and two come out. Front count is no longer a decreasing quantity, and finiteness of the number of
interactions needs a different argument (H&R Ch. 2). Empirically:
"""),
        code(r'''
def counts(f, nodes, jumps, T, seed, samples=300):
    rng = np.random.default_rng(seed)
    flux = DiscreteFlux(f, np.linspace(-1, 1, nodes))
    tr = FrontTracker.from_values(flux, np.sort(rng.uniform(-3, 3, jumps)),
                                  rng.uniform(-1, 1, jumps + 1), record_history=False)
    N0, N, E = len(tr.fronts), [], []
    for t in np.linspace(0, T, samples):
        tr.run(t)
        N.append(len(tr.fronts)); E.append(tr.event_count)
    return N0, np.array(N), np.array(E)

fig, ax = plt.subplots(figsize=(6.2, 4.4))
for name, f, c in (("Burgers", burgers, C_POLY), ("sin 3u", sin3, C_ENV)):
    N0, N, E = counts(f, 81, 40, T=10, seed=3)
    ax.plot(N0 - N, E, "-", color=c, label=f"{name}: N(0)={N0}, E(10)={E[-1]}")
m = ax.get_xlim()[1]
ax.plot([0, m], [0, m], ":", color=C_EXACT, lw=1, label="E = N(0) − N(t)")
ax.set(xlabel="fronts removed, N(0) − N(t)", ylabel="events E(t)",
       title="Events vs. fronts removed, t ∈ [0, 10]")
ax.legend(frameon=False)
plt.show()
'''),
        md(r"""
For Burgers the curve lies on the diagonal: every event removed exactly one front. For
$\sin 3u$ it rises above it, since some events leave the front count unchanged. The event count
is still finite, but it is no longer bounded by $N(0)$ alone. The total work is
$O(N\cdot E)$ with the scan of §1.
"""),
        md(r"""
## 7. Try it yourself

Random step data, any flux. Watch how the event count responds to the number of nodes (finer
staircases mean more fronts to absorb) and to nonconvexity. The sliders work only in
Jupyter or Colab.
"""),
        code(r'''
import ipywidgets as widgets
fluxes = {"Burgers u²/2": burgers, "cubic u³": lambda u: u**3, "sin(3u)": sin3,
          "traffic u(1-u)": lambda u: u * (1 - u)}
controls = dict(
    name=widgets.Dropdown(options=list(fluxes), value="sin(3u)", description="flux"),
    nodes=widgets.IntSlider(value=21, min=3, max=101, step=2, description="nodes"),
    jumps=widgets.IntSlider(value=10, min=1, max=40, description="jumps"),
    T=widgets.FloatSlider(value=6, min=0.5, max=20, step=0.5, description="T"),
    seed=widgets.IntSlider(value=0, min=0, max=20, description="seed"))
def show(name, nodes, jumps, T, seed):
    rng = np.random.default_rng(seed)
    flux = DiscreteFlux(fluxes[name], np.linspace(-1, 1, nodes))
    tr = FrontTracker.from_values(flux, np.sort(rng.uniform(-3, 3, jumps)),
                                  rng.uniform(-1, 1, jumps + 1))
    N0 = len(tr.fronts)
    tr.run(T)
    fig, ax = plt.subplots(figsize=(10, 4))
    xt_plot(ax, tr.segments, title=f"{name}: N(0)={N0}, N(T)={len(tr.fronts)}, "
                                   f"events={tr.event_count}")
    plt.show()
display(widgets.HBox(list(controls.values()), layout=widgets.Layout(flex_flow="row wrap")),
        widgets.interactive_output(show, controls))
'''),
        md(r"""
---
**Next:** *05 · Convergence*. As $\delta\to0$ and the initial steps refine, do these
polygonal solutions converge to the entropy solution, and how fast?
"""),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "04_interactions.ipynb", "toc_visible": True},
    })
    return nb


def convergence_notebook():
    cells = [
        md(r"""
# 05 · Convergence as δ → 0

Notebooks 03 and 04 built the method. For a fixed polygonal flux $f_\delta$ and step data
$u^\delta_0$, front tracking computes the **exact** entropy solution $u^\delta$ of
$$
u_t + f_\delta(u)_x = 0,\qquad u(x,0) = u^\delta_0(x).
$$
Every approximation is therefore made *before* the first event: $f$ is replaced by $f_\delta$,
and $u_0$ by $u_0^\delta$. After that the computation is exact (up to floating point, §7).
This notebook asks how far $u^\delta(\cdot,t)$ is from the entropy solution $u(\cdot,t)$ (in
Kružkov's sense) of the original problem, for general data rather than a single jump. The method
goes back to Dafermos (1972).

The answer depends on how the error is measured. Besides the usual $L^1$ distance we use the
**primitive** $U(x,t)=\int_{-\infty}^x u(y,t)\,dy$, the mass of $u$ to the left of $x$ (the precise
definition, and a table of all notation, are at the start of §1):

| error | rate | status |
|---|---|---|
| $\lVert u-u^\delta\rVert_{L^1}$ | $O(\delta)$ | theorem; cannot be improved |
| $\sup_x \lvert U-U^\delta\rvert$ | $O(\delta^2)$ | theorem, **if** $u_0^\delta$ conserves mass cell by cell |

The second line is the fundamental one, and the first follows from it (§1). It also explains
the $O(\delta^2)$ shock position seen in notebook 04 (§5).

1. Three estimates · 2. Projecting the initial data · 3. Burgers against an exact solution ·
4. Nearest-node rounding · 5. Shock positions · 6. A nonconvex flux · 7. Roundoff ·
8. Try it yourself

*Main reference: S. S. Ghoshal and J. D. Towers, "A convergence rate result for front tracking
approximations of conservation laws with discontinuous flux", arXiv:2509.22952 (2025). Our
setting is their case $f=g$, a flux without spatial discontinuity. Full references at the end.*
"""),
        code(SETUP),
        code(STYLE),
        md(r"""
## 1. Three estimates

Assume $f\in C^2$, $u_0\in BV$ with values in $[a,b]$, and $u_0$ constant outside $[-X,X]$.

**Notation.** A superscript $\delta$ always means "front tracking", a subscript $0$ always means
"at time $0$", and a capital letter always means "primitive of the lower-case function".

| symbol | meaning | in the code |
|---|---|---|
| $u(x,t)$ | exact entropy solution, flux $f$ | `burgers_exact(x, t)[0]` |
| $u^\delta(x,t)$ | front tracking solution, flux $f_\delta$ | `tr.sample(x)` |
| $u_0(x)=u(x,0)$ | exact initial data | `u0(x)` |
| $u_0^\delta(x)=u^\delta(x,0)$ | projected (step) initial data, §2 | a tracker before `run` |
| $u_L$ | far-left state, the same for all four ($0$ in our examples) | |
| $U(x,t)=\displaystyle\int_{-\infty}^x\big(u(y,t)-u_L\big)\,dy$ | primitive of $u$: mass to the left of $x$ | `burgers_exact(x, t)[1]` |
| $U^\delta(x,t)=\displaystyle\int_{-\infty}^x\big(u^\delta(y,t)-u_L\big)\,dy$ | primitive of $u^\delta$ | `primitive(tr, x)` |
| $U_0(x)=U(x,0)$, $U_0^\delta(x)=U^\delta(x,0)$ | primitives of the initial data | `U0(x)`, `primitive(tr, x)` |
| $\sup_x\lvert U-U^\delta\rvert$ | **primitive error**: the largest net mass misplaced across any point $x$ | column `prim err` in tables |
| $w_0<\dots<w_K$, $\delta$ | state nodes of $f_\delta$, and their largest spacing | `flux.u_grid` |

Subtracting $u_L$ only makes the integrals finite; it cancels in every difference $U-U^\delta$. For our
data $u_L=0$, so $U$ is literally "the mass to the left of $x$". The polygonal flux $f_\delta$
interpolates $f$ at the nodes: $f_\delta(w_k)=f(w_k)$, linear in between, with $w_0=a$, $w_K=b$,
$w_{k+1}-w_k\le\delta$. (Nodes are called $w_k$, not $u_k$, so they cannot be confused with $u_0$.)

**(E1) Interpolation.** $\;\lVert f-f_\delta\rVert_\infty\le\tfrac18\lVert f''\rVert_\infty\,\delta^2.$
The argument is cell by cell. On an open cell $(w_k,w_{k+1})$ the interpolant is linear, so
$e=f-f_\delta$ is $C^2$ there with $e''=f''$, and $e$ vanishes at both nodes. Hence
$e(u)=-\tfrac12 f''(\xi)\,(u-w_k)(w_{k+1}-u)$ for some $\xi$ in the cell, and
$(u-w_k)(w_{k+1}-u)\le\delta^2/4$. As a distribution, $f_\delta''$ has point masses at the nodes, but
the argument never differentiates across a node, so they do not enter. (Standard; Ghoshal–Towers,
Lemma 3.7.)

**(E2) Stability of primitives.**
$$
\lVert U(t)-U^\delta(t)\rVert_\infty \;\le\; \lVert U_0-U_0^\delta\rVert_\infty + t\,\lVert f-f_\delta\rVert_\infty .
$$
No convexity is needed. (Ghoshal–Towers, Lemma 3.1.) The primitive solves the Hamilton–Jacobi
equation $U_t+f(U_x)=0$, and the sup norm is the natural contraction norm for Hamilton–Jacobi.
Karlsen and Risebro (2002) prove, through front tracking, that viscosity solutions of that
equation are exactly the primitives of entropy solutions. The sup norm of $U-U^\delta$ is also
closely tied to the 1-Wasserstein distance, in which Solem (2018) proved an $O(\delta^2)$ rate for convex $f$.

**(E3) From primitives to $L^1$.** Let $w=u-u^\delta$ vanish outside $[-Y,Y]$, carry no net
mass, and let both solutions have total variation at most $K$. Then
$$
\lVert w\rVert_{L^1}\le 2\big(Y K\,\lVert W\rVert_\infty\big)^{1/2},\qquad W=U-U^\delta .
$$
*Proof.* By Cauchy–Schwarz and integration by parts ($W(\pm Y)=0$),
$\big(\int|w|\big)^2\le 2Y\!\int w^2 = 2Y\!\int w\,dW = -2Y\!\int W\,dw\le 2Y\lVert W\rVert_\infty\,\mathrm{TV}(w)$,
and $\mathrm{TV}(w)\le 2K$. $\square$ (This is Ghoshal–Towers' argument, (4.3)–(4.7).)

**Putting them together.** §2 constructs data with $\lVert U_0-U_0^\delta\rVert_\infty\le\delta^2$.
Then E1 and E2 give
$$
\lVert U(t)-U^\delta(t)\rVert_\infty\le C_1\delta^2,\qquad C_1=1+\tfrac t8\lVert f''\rVert_\infty ,
$$
and E3, with $K=\mathrm{TV}(u_0)$ and $Y=X+t\,\mathrm{Lip}(f)$, gives
$$
\lVert u(t)-u^\delta(t)\rVert_{L^1}\le 2\big(Y\,\mathrm{TV}(u_0)\,C_1\big)^{1/2}\,\delta .
$$
This is Ghoshal–Towers, Theorem 4.3, with a smaller $Y$. Their $Y=X+2t\,\mathrm{Lip}(f)$ comes from
the time-step restriction of the Godunov scheme used in their proof. Finite speed of propagation
gives $X+t\,\mathrm{Lip}(f)$ directly, for $u$ and for $u^\delta$ alike, since the slopes of $f_\delta$ are
bounded by $\mathrm{Lip}(f)$. First-order convergence in $L^1$ is classical (Lucier, 1986).
Their route is different and makes the $\delta^2$ primitive estimate the source of it.

$O(\delta)$ in $L^1$ cannot be improved: $u^\delta$ only takes grid values, so even where $u$ is smooth
the error is about $\delta/4$ per unit length. Monotone finite-volume schemes are worse still. Their
general $L^1$ rate is $\delta^{1/2}$ (Kuznetsov, 1976), and this is sharp (Şabac, 1997). Front tracking
does better because it adds no numerical viscosity: $u^\delta$ is an exact entropy solution of a
nearby problem.

The rest of the notebook measures both rates and checks that the observed errors stay below
the bounds.
"""),
        md(r"""
## 2. Projecting the initial data

Front tracking needs step data $u_0^\delta$ whose values are grid nodes. How we build it from $u_0$
decides the initial primitive error. Recall the two primitives at $t=0$:
$$
U_0(x)=\int_{-\infty}^x\big(u_0(y)-u_L\big)\,dy,\qquad U_0^\delta(x)=\int_{-\infty}^x\big(u_0^\delta(y)-u_L\big)\,dy .
$$
Their difference $U_0^\delta(x)-U_0(x)$ is the net mass that the projection has moved from the right
of $x$ to its left. E2 says this initial error is carried forward in time without growing. So we want
it small **at every $x$**, not just on average.

The library's `FrontTracker.from_values` samples $u_0$ and rounds each value to the nearest
node. That is cheap, but it does not conserve mass locally. Each cell makes an error of up to
$\delta/2$ in $u$, and on a stretch where these errors share a sign they add up in
$U_0^\delta-U_0$. Rounding can even increase the total variation.

Ghoshal and Towers (their (3.35)–(3.38)) use a different projection:

* partition $[-X,X]$ into cells of width $\le\delta$ such that $\mathrm{TV}(u_0)\le\delta$ on each
  open cell (so every jump larger than $\delta$ sits on a cell edge);
* take the **exact cell average** on each cell.

Then $U_0^\delta=U_0$ at every cell edge, because each cell carries exactly its own mass. Inside a
cell $|U_0^\delta-U_0|\le(\text{width})\cdot\mathrm{TV}\le\delta^2$ (their Lemma 3.7). Averaging also
gives $\mathrm{TV}(u_0^\delta)\le\mathrm{TV}(u_0)$.

The averages are not grid nodes. `fronttrack` needs node values, so we **add the averages to the
state grid**. The grid stays sorted with spacing $\le\delta$, and $f_\delta$ still interpolates $f$,
so E1 is unchanged. `DiscreteFlux` already accepts nonuniform grids.

Our test data for Burgers' flux $f(u)=u^2/2$ are a bump next to a block:
$$
u_0(x)=A\,(1-x^2)_+ + B\,\mathbf 1_{(1,\,1.5)}(x),\qquad A=0.68,\;B=0.6 .
$$
The block gives a jump up (a rarefaction) and a jump down (a shock) at $t=0$. $A$ is chosen so that
the top of the bump is not a grid value for any $\delta$ used below.

For Burgers the entropy solution is known exactly through the Hopf–Lax formula
$U(x,t)=\min_y\big[U_0(y)+\tfrac{(x-y)^2}{2t}\big]$, with $u=(x-y^*)/t$. The code minimises over a
finite candidate set that provably contains the minimiser, so the reference involves no grid search
and no tolerance. Primitives of step functions are computed exactly (they are piecewise linear).
"""),
        code(r'''
burgers = lambda u: 0.5 * u**2
A, B = 0.68, 0.6            # bump height (not a grid value), block height (a grid value)

def u0(x, A=A, B=B):
    """A(1-x^2) on [-1,1], plus a block of height B on (1, 1.5); zero elsewhere."""
    x = np.asarray(x, dtype=float)
    return A * np.clip(1 - x**2, 0, None) + B * ((x > 1) & (x < 1.5))

def U0(x, A=A, B=B):
    """Primitive of u0 from -infinity (u0 = 0 far left)."""
    y = np.clip(x, -1, 1)
    return A * (y - y**3 / 3 + 2 / 3) + B * np.clip(np.asarray(x) - 1, 0, 0.5)

def burgers_exact(x, t, A=A, B=B):
    """Exact entropy solution for Burgers and the data above, via Hopf–Lax:
    U(x,t) = min_y [U0(y) + (x-y)^2/(2t)],  u(x,t) = (x - y*)/t.
    Every y gives an upper bound, so it is enough to minimise over a candidate set
    that surely contains the minimiser: the stationary points y + t u0(y) = x of each
    smooth piece of u0, and the jump points of u0. No grid search, no tolerance."""
    x = np.asarray(x, dtype=float)
    cands = [x, x - B * t, np.full_like(x, -1.0), np.full_like(x, 1.0), np.full_like(x, 1.5)]
    disc = 1 - 4 * t * A * (x - t * A)               # t A y^2 - y + (x - t A) = 0
    root = np.sqrt(np.clip(disc, 0, None))
    for sign in (1, -1):
        y = np.clip((1 + sign * root) / (2 * t * A), -1, 1)
        cands.append(np.where(disc >= 0, y, x))
    Y = np.array(cands)
    vals = U0(Y, A, B) + (x - Y)**2 / (2 * t)
    k = np.argmin(vals, axis=0)
    y_star = Y[k, np.arange(x.size)]
    return (x - y_star) / t, vals[k, np.arange(x.size)], y_star

def mass_projection(f, U0, breaks, delta, lip, grid_lo, grid_hi, u_out=0.0):
    """Ghoshal–Towers projection. Cells of width <= delta and TV(u0) <= delta on each
    open cell (width delta/lip where |u0'| <= lip; jumps only at `breaks`, which become
    cell edges). States are exact cell averages, added to the state grid."""
    h = delta / max(1.0, lip)
    edges = [breaks[0]]
    for a, b in zip(breaks[:-1], breaks[1:]):
        edges.extend(np.linspace(a, b, int(np.ceil((b - a) / h)) + 1)[1:])
    edges = np.array(edges)
    avg = np.diff(U0(edges)) / np.diff(edges)
    nodes = np.linspace(grid_lo, grid_hi, int(round((grid_hi - grid_lo) / delta)) + 1)
    grid = np.unique(np.concatenate([nodes, avg]))
    grid = grid[np.concatenate([[True], np.diff(grid) > 1e-12])]   # merge roundoff twins
    flux = DiscreteFlux(f, grid)
    states = flux.state_index(np.concatenate([[u_out], avg, [u_out]]))
    return FrontTracker(flux, edges, states, record_history=False)

def nearest_projection(f, u0, lo, hi, delta, grid_lo, grid_hi, u_out=0.0):
    """The library's default: sample u0 at cell midpoints (width delta), round to nodes."""
    edges = np.linspace(lo, hi, int(round((hi - lo) / delta)) + 1)
    nodes = np.linspace(grid_lo, grid_hi, int(round((grid_hi - grid_lo) / delta)) + 1)
    values = np.concatenate([[u_out], u0(0.5 * (edges[1:] + edges[:-1])), [u_out]])
    return FrontTracker.from_values(DiscreteFlux(f, nodes), edges, values,
                                    record_history=False)

def primitive(tr, x):
    """Exact primitive of the step solution, int_{-inf}^x (u - u_left): piecewise linear."""
    g = tr.flux.u_grid
    p = np.array([fr.x for fr in tr.fronts])
    v = g[[tr.left_state] + [fr.iR for fr in tr.fronts]] - g[tr.left_state]
    if p.size == 0:
        return np.zeros_like(np.asarray(x, dtype=float))
    cum = np.concatenate([[0.0], np.cumsum(np.diff(p) * v[1:-1])])
    k = np.searchsorted(p, x, side="right")
    j = np.maximum(k - 1, 0)
    return np.where(k == 0, 0.0, cum[j] + (x - p[j]) * v[k])

def front_positions(tr):
    return np.array([fr.x for fr in tr.fronts])

def exact_shocks(t, A=A, B=B, lo=-1.5, hi=8.0, n=200001):
    """Exact shock locations: y*(x) is nondecreasing and jumps across a shock.
    Bracket each jump on a grid, then bisect to machine precision."""
    x = np.linspace(lo, hi, n)
    _, _, y = burgers_exact(x, t, A, B)
    out = []
    for i in np.nonzero(np.diff(y) > 1e-6 + 10 * (x[1] - x[0]))[0]:
        a, b = x[i], x[i + 1]
        ya, yb = y[i], y[i + 1]
        for _ in range(100):
            m = 0.5 * (a + b)
            if m in (a, b):
                break
            ym = burgers_exact(np.array([m]), t, A, B)[2][0]
            if ym - ya < yb - ym:
                a, ya = m, ym
            else:
                b, yb = m, ym
        out.append(0.5 * (a + b))
    return np.array(out)

d = 0.1
x = np.linspace(-1.3, 1.8, 3001)
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
a.plot(x, u0(x), color=C_EXACT, lw=1.5, label="$u_0$")
for proj, c in (("mass", C_POLY), ("nearest", C_ENV)):
    tr = (mass_projection(burgers, U0, [-1.0, 1.0, 1.5], d, 2 * A, 0.0, 1.0) if proj == "mass"
          else nearest_projection(burgers, u0, -1.0, 1.5, d, 0.0, 1.0))
    a.step(x, tr.sample(x), where="post", color=c, lw=1.3, label=proj)
    b.plot(x, primitive(tr, x) - U0(x), color=c, lw=1.5, label=proj)
a.set(xlabel="$x$", title=f"initial data, δ = {d}")
b.set(xlabel="$x$", title=r"$U_0^\delta - U_0$")
a.legend(frameon=False); b.legend(frameon=False)
plt.show()
'''),
        md(r"""
Left: the two step approximations $u_0^\delta$. Right: the mass each one misplaces, $U_0^\delta(x)-U_0(x)$.
With nearest-node rounding the misplaced mass drifts with one sign across the bump and the block.
With cell averages it returns to zero at every cell edge.

## 3. Burgers against an exact solution

We run both rates at $T=3$, after the bump has formed its shock, for $\delta=0.1\cdot2^{-k}$. The
bounds of §1 use $\lVert f''\rVert_\infty=1$, $\mathrm{Lip}(f)=1$ on $[0,1]$, $X=1.5$ (so $Y=4.5$) and
$\mathrm{TV}(u_0)=2A+2B$.
"""),
        code(r'''
T = 3.0
deltas = 0.1 / 2 ** np.arange(6)
x_dense = np.linspace(-1.5, 6.0, 400_001)
shocks = exact_shocks(T)

def errors(tr, T, A=A, B=B, shocks=shocks):
    """L1 error (dense quadrature) and exact sup of |U - U_delta| over dense points,
    every front and every exact shock."""
    xs = np.sort(np.concatenate([x_dense, front_positions(tr), shocks]))
    u_ex, U_ex, _ = burgers_exact(xs, T, A, B)
    L1 = np.trapezoid(np.abs(tr.sample(xs) - u_ex), xs)
    return L1, np.max(np.abs(primitive(tr, xs) - U_ex))

C1, Y, TV0 = 1 + T / 8, 1.5 + T, 2 * A + 2 * B
res = {"mass": [], "nearest": []}
for d in deltas:
    res["mass"].append(errors(mass_projection(burgers, U0, [-1.0, 1.0, 1.5], d, 2 * A,
                                              0.0, 1.0).run(T), T))
    res["nearest"].append(errors(nearest_projection(burgers, u0, -1.0, 1.5, d,
                                                    0.0, 1.0).run(T), T))
res = {k: np.array(v) for k, v in res.items()}
L1, P = res["mass"].T

# Outputs are collected and printed once per cell, so that the saved notebook does not
# depend on when the kernel happened to flush stdout.
lines = [f"{'delta':>9} {'L1':>10} {'rate':>5} {'L1/bound':>9} {'prim err':>10} {'rate':>5}"
         f" {'/delta^2':>8} {'/bound':>7}"]
for i, d in enumerate(deltas):
    r1 = "" if i == 0 else f"{np.log2(L1[i-1] / L1[i]):5.2f}"
    r2 = "" if i == 0 else f"{np.log2(P[i-1] / P[i]):5.2f}"
    lines.append(f"{d:9.6f} {L1[i]:10.3e} {r1:>5} {L1[i] / (2 * np.sqrt(Y * TV0 * C1) * d):9.3f}"
                 f" {P[i]:10.3e} {r2:>5} {P[i] / d**2:8.4f} {P[i] / (C1 * d**2):7.3f}")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.loglog(deltas, L1, "o-", color=C_POLY, ms=5, label=r"$\|u-u^\delta\|_{L^1}$")
ax.loglog(deltas, P, "s-", color=C_ENV, ms=5, label=r"$\sup|U-U^\delta|$")
ax.loglog(deltas, 2 * np.sqrt(Y * TV0 * C1) * deltas, ":", color=C_POLY, lw=1.2,
          label="bound (E3)")
ax.loglog(deltas, C1 * deltas**2, ":", color=C_ENV, lw=1.2, label="bound (E2)")
ax.set(xlabel=r"$\delta$", title=f"Burgers, bump + block, T = {T:g}")
ax.legend(frameon=False)
plt.show()
'''),
        md(r"""
Both rates are as predicted: $1$ in $L^1$ and $2$ for the primitive. Both ratios to the bounds
stay below $1$. The primitive bound is within a factor of four. The $L^1$ bound is loose by a factor
of about 20, and most of that comes from E3 itself: with the *observed* primitive error in place of its
bound, E3 still overestimates by about 10. Cauchy–Schwarz spreads the error evenly over the whole
support, and $\mathrm{TV}(w)\le2K$ ignores cancellation.

### How the primitive error grows in time

For this flux ($\lVert f''\rVert_\infty=1$) and this projection, E1 and E2 bound the primitive error at time
$t$ by the sum of two terms:
$$
\sup_x|U-U^\delta|(t)\;\le\;\underbrace{\sup_x|U_0-U_0^\delta|}_{\text{error in the data, }\le\,\delta^2}
\;+\;\underbrace{t\,\lVert f-f_\delta\rVert_\infty}_{\text{error from the flux, }\le\,\frac t8\delta^2}.
$$
The first term is present at $t=0$ and E2 says it never grows. The second starts at zero and grows
linearly: at every instant the polygonal flux moves mass at a rate that differs from the true rate by
at most $\lVert f-f_\delta\rVert_\infty$.

Both terms carry a factor $\delta^2$, so we divide by it. The quantity plotted below is
$$
R(t)=\frac{\sup_x|U(x,t)-U^\delta(x,t)|}{\delta^2},\qquad\text{and E1, E2 say}\quad R(t)\le 1+\frac t8 .
$$
If the theory has the right order in $\delta$, $R(t)$ should hardly change when $\delta$ is divided by four.
The cell computes $R$ at nine times for $\delta=0.0125$ and $\delta=0.003125$.
"""),
        code(r'''
times = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
x_wide = np.linspace(-1.5, 8.0, 400_001)
fig, ax = plt.subplots(figsize=(6.5, 4))
lines = []
for d, c in ((0.0125, C_POLY), (0.003125, C_ENV)):
    tr = mass_projection(burgers, U0, [-1.0, 1.0, 1.5], d, 2 * A, 0.0, 1.0)
    ratio = []
    for t in times:
        tr.run(t)
        xs = np.sort(np.concatenate([x_wide, front_positions(tr)]))
        if t == 0:
            U_ex = U0(xs)
        else:
            xs = np.sort(np.concatenate([xs, exact_shocks(t)]))
            U_ex = burgers_exact(xs, t)[1]
        ratio.append(np.max(np.abs(primitive(tr, xs) - U_ex)) / d**2)
    ax.plot(times, ratio, "o-", color=c, ms=4, label=f"δ = {d}")
    lines.append(f"delta={d}: " + "  ".join(f"t={t:g}: {r:.3f}" for t, r in zip(times, ratio)))
print("\n".join(lines))
R0 = ratio[0]                                   # observed data error, finer delta
ax.plot(times, 1 + times / 8, ":", color=C_EXACT, lw=1.2, label="proved bound 1 + t/8")
ax.plot(times, R0 + times / 8, "-.", color=C_EXACT, lw=1, alpha=0.6,
        label=f"observed R(0) + t/8 = {R0:.2f} + t/8")
ax.plot(times, times / 8, "--", color=C_FLUX, lw=1.2, label="flux term alone, t/8")
ax.set(xlabel="$t$", ylabel=r"$R(t)=\sup|U-U^\delta|\,/\,\delta^2$",
       title="Primitive error in time, in units of δ²", ylim=(-0.05, 2.6))
ax.legend(frameon=False, loc="upper left")
plt.show()
'''),
        md(r"""
How to read the plot, from left to right:

1. **At $t=0$**, $R(0)\approx0.09$ for both $\delta$. This is the data error alone: the projection
   misplaces about $0.09\,\delta^2$ of mass at worst, well below the $\delta^2$ that Lemma 3.7 allows.
2. **Up to $t\approx0.5$**, $R(t)$ stays at that value. The flux term $t/8$ (grey dashed line) is still
   smaller than the data error, and the data error is carried along without growing, as E2 says.
3. **From $t\approx1$ on**, $R(t)$ grows linearly and runs just below the dashed line $t/8$. Now the
   flux term is the larger one, and it is what we see.
4. **The two terms do not add up.** If they did, the curves would follow the dash-dotted line
   $R(0)+t/8$. From $t\approx1$ on they stay below even $t/8$. So in this example the error behaves like the **larger** of
   the two terms, not their sum. The proved bound (dotted) is the sum, which is why it is loose by
   about the data term.
5. **The constant $\tfrac18$ is nearly sharp.** When $\delta$ is divided by four, the late-time values move
   up towards $t/8$ (at $t=6$: $0.62$, then $0.68$, against $0.75$). So the factor $\tfrac18\lVert f''\rVert_\infty$ in
   E1 and E2 cannot be improved much for these data.

Points 4 and 5 are observations for one data set, not theorems. We have not worked out which part of
the solution attains the flux term.

## 4. Nearest-node rounding

Now the same experiment with the library's default projection. The first figure of §2 predicts
the problem: its initial primitive error is already of order $\delta$ in places.
"""),
        code(r'''
Ln, Pn = res["nearest"].T
lines = [f"{'delta':>9} {'L1 mass':>10} {'L1 nearest':>11} {'sup mass':>10} {'sup nearest':>12}"
         f" {'nearest/delta^2':>16}"]
for i, d in enumerate(deltas):
    lines.append(f"{d:9.6f} {L1[i]:10.3e} {Ln[i]:11.3e} {P[i]:10.3e} {Pn[i]:12.3e}"
                 f" {Pn[i] / d**2:16.2f}")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.loglog(deltas, P, "o-", color=C_POLY, ms=5, label="mass-preserving")
ax.loglog(deltas, Pn, "s-", color=C_ENV, ms=5, label="nearest node")
ax.loglog(deltas, deltas**2, ":", color=C_EXACT, lw=1, label=r"$\delta^2$")
ax.loglog(deltas, 0.05 * deltas, "--", color=C_FLUX, lw=1, label=r"$\propto\delta$")
ax.set(xlabel=r"$\delta$", ylabel=r"$\sup|U-U^\delta|$", title=f"Primitive error at T = {T:g}")
ax.legend(frameon=False)
plt.show()
'''),
        md(r"""
In $L^1$ both projections are first order, and nearest-node rounding costs only a factor of about
two. The primitive is a different story. With nearest-node rounding its error is erratic, is 6 to
40 times the mass-preserving one, and has no clean rate. It depends on
where the grid values happen to fall relative to the data.

A heuristic for the size of the error: rounding errors have one sign wherever $u_0$ stays within a
single grid cell. Near a smooth maximum at a non-grid value that stretch has length about
$\sqrt{\delta/|u_0''|}$, giving an error of about $\delta^{3/2}$. On a plateau at a non-grid value the
stretch has fixed length, giving an error of order $\delta$. Neither is $\delta^2$.

The practical lesson for anyone using `fronttrack`: if you care about *where* things are (shock
positions, mass distribution), project the data with cell averages, not by rounding.

## 5. Shock positions

Suppose $u$ has an isolated shock of strength $[u]$ at $x_s$, and $u^\delta$ has it at $x_s+e$. Just
beside the shock, $U^\delta-U\approx\pm[u]\,e$. This suggests
$$
|e|\;\lesssim\;\frac{\sup|U-U^\delta|}{[u]} = O(\delta^2).
$$
This is a heuristic corollary of E2, not a proved statement: it ignores the $O(\delta)$ differences in
the states next to the shock. In notebook 04 the data were node-valued, so $U_0^\delta=U_0$ exactly and
only the flux term $\tfrac t8\delta^2$ was present. That is why the shock position there came out at
$O(\delta^2)$.

Test: the bump alone ($B=0$), which forms a single shock. The approximate shock is the front with
the largest jump.
"""),
        code(r'''
T5 = 3.0
xs_exact = exact_shocks(T5, A, 0.0)[0]
u_left = burgers_exact(np.array([xs_exact - 1e-9]), T5, A, 0.0)[0][0]
lines = [f"exact shock at x = {xs_exact:.12f}, strength [u] = {u_left:.6f}"]
bump_U0 = lambda z: U0(z, A, 0.0)
bump_u0 = lambda z: u0(z, A, 0.0)

def shock_front(tr):
    g = tr.flux.u_grid
    return max(tr.fronts, key=lambda f: abs(g[f.iR] - g[f.iL])).x

deltas5 = 0.1 / 2 ** np.arange(7)
pos = {"mass": [], "nearest": []}
for d in deltas5:
    pos["mass"].append(shock_front(mass_projection(burgers, bump_U0, [-1.0, 1.0], d, 2 * A,
                                                   0.0, 1.0).run(T5)))
    pos["nearest"].append(shock_front(nearest_projection(burgers, bump_u0, -1.0, 1.0, d,
                                                         0.0, 1.0).run(T5)))
err = {k: np.abs(np.array(v) - xs_exact) for k, v in pos.items()}
bound = (1 + T5 / 8) * deltas5**2 / u_left
lines.append(f"{'delta':>9} {'|e| mass':>10} {'|e|/d^2':>8} {'|e| nearest':>12}"
             "  heuristic C1 d^2/[u]")
for i, d in enumerate(deltas5):
    lines.append(f"{d:9.6f} {err['mass'][i]:10.2e} {err['mass'][i] / d**2:8.3f}"
                 f" {err['nearest'][i]:12.2e}  {bound[i]:.2e}")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.loglog(deltas5, err["mass"], "o-", color=C_POLY, ms=5, label="mass-preserving")
ax.loglog(deltas5, err["nearest"], "s-", color=C_ENV, ms=5, label="nearest node")
ax.loglog(deltas5, bound, ":", color=C_EXACT, lw=1.2, label=r"$C_1\delta^2/[u]$")
ax.set(xlabel=r"$\delta$", ylabel="shock position error", title="Bump, single shock, T = 3")
ax.legend(frameon=False)
plt.show()
'''),
        md(r"""
With cell averages the shock position error stays below $C_1\delta^2/[u]$, by a factor of 30 or
more, and falls at roughly second order. It is not monotone, because the front with the largest jump changes as the staircase
next to the shock is absorbed one step at a time. With nearest-node rounding the error is again
irregular and much larger.

## 6. A nonconvex flux

E2 does not use convexity, so the $\delta^2$ primitive estimate holds for $f(u)=u^3$ too. There is no
Hopf–Lax formula here, so the reference is front tracking itself at $\delta_{\rm ref}=0.0125/8$, with the
same projection. This is a **self-convergence** test. The reference has its own error, about
$1/64$ of the finest error below in the primitive and $1/8$ in $L^1$. Between two step functions both
errors are computed exactly: $L^1$ over the merged breakpoints, and the primitive difference at the
fronts, where its maximum is attained (both primitives are piecewise linear).

Data: $u_0(x)=0.8\sin(\pi x)$ on $[-1,1]$, zero outside, at $T=1$. The states cross the inflection
point $u=0$, so the Riemann problems produce compound waves. Here $\lVert f''\rVert_\infty=4.8$ on
$[-0.8,0.8]$.
"""),
        code(r'''
import math
cubic = lambda u: u**3
sine_U0 = lambda z: 0.8 / np.pi * (-np.cos(np.pi * np.clip(z, -1, 1)) - 1)
T6, lip6 = 1.0, 0.8 * np.pi
run6 = lambda d: mass_projection(cubic, sine_U0, [-1.0, 1.0], d, lip6, -0.8, 0.8).run(T6)

def step_errors(a, b):
    p = np.unique(np.concatenate([front_positions(a), front_positions(b)]))
    mid = 0.5 * (p[1:] + p[:-1])
    L1 = math.fsum(np.abs(a.sample(mid) - b.sample(mid)) * np.diff(p))
    return L1, np.max(np.abs(primitive(a, p) - primitive(b, p)))

ref = run6(0.0125 / 8)
deltas6 = 0.1 / 2 ** np.arange(4)
e6 = np.array([step_errors(run6(d), ref) for d in deltas6])
C1_6 = 1 + T6 * 4.8 / 8
lines = [f"reference: {len(ref.fronts)} fronts, {ref.event_count} events",
         f"{'delta':>8} {'L1':>10} {'L1/delta':>9} {'prim err':>10} {'/delta^2':>9} {'/bound':>7}"]
for d, (l1, p) in zip(deltas6, e6):
    lines.append(f"{d:8.4f} {l1:10.3e} {l1 / d:9.3f} {p:10.3e} {p / d**2:9.4f}"
                 f" {p / (C1_6 * d**2):7.3f}")
print("\n".join(lines))
'''),
        md(r"""
Both error constants are bounded, and the primitive stays well inside the bound. The constants are not
monotone in $\delta$: $\delta=0.025$ is visibly worse than its neighbours. A likely cause is the one seen
in notebook 03, §3. Where a compound wave's tangency point falls between grid nodes, the discrete
shock speed is off by $O(\delta)$, and how far off depends on where the nodes fall. We have not checked
this. The estimates only promise an upper bound, and it holds.

## 7. Roundoff in the collinearity test

`solve_riemann` keeps a node $b$ between $a$ and $c$ only if its vertical gap from the chord $a\to c$
lies on the envelope side by more than $16\varepsilon S$. (The gap is an orientation test divided by a
positive length; Shewchuk (1997) analyses such tests.) Here $\varepsilon=2^{-52}$ and
$$
S=\max(|f_a|,|f_b|,|f_c|)+|s_{ac}|\max(|w_0|,|w_K|),\qquad s_{ac}=\frac{f_c-f_a}{w_c-w_a}.
$$
The constant 16 was calibrated, not derived. Here is the derivation.

**Arithmetic.** The code evaluates
$g=(f_b-f_a)-(f_c-f_a)\cdot\big((w_b-w_a)/(w_c-w_a)\big)$ with seven floating-point operations. In the
standard model $\mathrm{fl}(x\circ y)=(x\circ y)(1+\theta)$, $|\theta|\le u=\varepsilon/2$, the quotient
$\lambda=(w_b-w_a)/(w_c-w_a)\in(0,1)$ carries a relative error of at most $3u$, and the product
$P=(f_c-f_a)\lambda$ at most $5u$. With $D=f_b-f_a$ and $|g|\le|D|+|P|$ this gives
$$
|\hat g-g|\le u|D|+5u|P|+u|g|+O(u^2)\le 2u|D|+6u|P|+O(u^2).
$$
Now $|D|\le 2\max|f|$, and $|P|=|s_{ac}|(w_b-w_a)\le 2|s_{ac}|\max(|w_0|,|w_K|)$. Hence
$$
|\hat g-g|\le 12u\,S = 6\varepsilon S\quad(+O(\varepsilon^2)),
$$
where $g$ is the exact gap of the stored numbers.

**The one hypothesis: how accurately $f$ was evaluated.** Suppose each stored value satisfies
$|\tilde f(w)-f(w)|\le k\,u\,|f(w)|$ ($k$ units of roundoff). The gap is linear in the three values with
coefficients $1,\,1-\lambda,\,\lambda$, so this adds at most $2ku\max|f|\le k\varepsilon S$. Relative to the
true gap $g^*$ of the polygon through $(w_i,f(w_i))$:
$$
|\hat g-g^*|\le(6+k)\,\varepsilon S .
$$

**Consequences for the test.**

* A node is kept only if its true gap lies on the envelope side by more than $(10-k)\varepsilon S$. For
  $k<10$, exactly collinear nodes are always merged, so there are no co-moving duplicate fronts.
* A node is dropped only if its true gap is less than $(22+k)\varepsilon S$. Genuine vertices beyond that
  are always kept.
* Only nodes in the band between these two can be misjudged.

**Size of the consequence (heuristic).** A misjudged node changes the envelope by less than
$(22+k)\varepsilon S$ in sup norm. By E2, the primitive of that one Riemann solution then moves by at most $t$
times this. Compare the discretisation term $\tfrac t8\lVert f''\rVert\delta^2$: the two are equal only near
$\delta\approx\sqrt{8(22+k)\varepsilon S/\lVert f''\rVert}\approx 10^{-7}$ for $S$ and $\lVert f''\rVert$ of order one.
Summed over many events this is a count-times-bound argument, not a theorem.

Below, both ingredients are measured. The arithmetic error is checked against exact rational
arithmetic (`fractions.Fraction`) on the stored doubles, over random triples and over the exactly
collinear triples of $u^3$ (pairs of nodes symmetric about a tangency point, as in 03 §4). The value of $k$ is
measured for `u**3`, where the exact cube of a double is also rational.
"""),
        code(r'''
from fractions import Fraction as Fr
eps = np.finfo(float).eps

def gap_float(w, v, a, b, c):
    """The library's test, operation for operation (riemann.py)."""
    rise, run = v[c] - v[a], w[c] - w[a]
    gap = (v[b] - v[a]) - rise * ((w[b] - w[a]) / run)
    S = max(abs(v[a]), abs(v[b]), abs(v[c])) + abs(rise / run) * max(abs(w[0]), abs(w[-1]))
    return gap, S

def gap_exact(w, v, a, b, c):
    W = [Fr(float(w[i])) for i in (a, b, c)]
    V = [Fr(float(v[i])) for i in (a, b, c)]
    return (V[1] - V[0]) - (V[2] - V[0]) * (W[1] - W[0]) / (W[2] - W[0])

def worst_ratio(triples, w, v):
    return max(float(abs(Fr(float(g)) - gap_exact(w, v, *t)) / Fr(float(eps * S)))
               for t in triples for g, S in [gap_float(w, v, *t)])

rng = np.random.default_rng(0)
tests = {"u³ on [-1,1]": (lambda u: u**3, -1, 1), "sin 3u": (lambda u: np.sin(3 * u), -1, 1),
         "u²/2 + 1000": (lambda u: 0.5 * u * u + 1000, -1, 1), "u³ on [0,1]": (lambda u: u**3, 0, 1)}
lines = ["worst |computed gap - exact gap| / (eps S)   (bound: 6)"]
for name, (f, lo, hi) in tests.items():
    worst = 0.0
    for n in (21, 201, 2001):
        w = np.linspace(lo, hi, n); v = f(w)
        worst = max(worst, worst_ratio([tuple(np.sort(rng.choice(n, 3, replace=False)))
                                        for _ in range(1000)], w, v))
    lines.append(f"  {name:14s} random triples: {worst:.3f}")

# Exactly collinear in the intended grid: for u^3 the chord slope from -1 is
# 1 - w_b + w_b^2, symmetric about 1/2, so nodes 1/2 - h and 1/2 + h are collinear with -1.
w = np.linspace(-1, 1, 2001); v = w**3
sym = [(0, b, 3000 - b) for b in range(1001, 1500)]   # w_b + w_c = 1
lines.append(f"  u³ collinear triples (0, 1/2-h, 1/2+h): {worst_ratio(sym, w, v):.3f}")
g_sym = max(abs(gap_float(w, v, *t)[0]) / (eps * gap_float(w, v, *t)[1]) for t in sym)
lines.append(f"  ...their computed |gap| / (eps S): {g_sym:.3f}   (merged if < 16)")

k = max(float(abs(Fr(float(fv)) - Fr(float(wv))**3) / (Fr(float(eps / 2)) * abs(Fr(float(wv))**3)))
        for fv, wv in zip(v, w) if wv != 0)
lines.append(f"measured k for numpy's u**3 on 2001 nodes: {k:.3f} units of roundoff")
print("\n".join(lines))
'''),
        md(r"""
The arithmetic error stays well under the proved $6\varepsilon S$. The collinear $u^3$ triples come out
with computed gaps of about $\varepsilon S$, far inside the merge threshold (their stored nodes are not
exactly symmetric, so their exact gap is not quite zero either). NumPy's `u**3` is accurate to about
one unit of roundoff, so $k\approx1$. The tolerance $16\varepsilon S$ therefore leaves a margin of about
$10-k\approx 9$ units on the merge side and needs genuine vertices to clear $23\varepsilon S$. The
genuine gaps measured in PR #3 were above $10^9\varepsilon S$.

The derivation assumes a flux that is evaluated accurately. A flux computed by a long formula
with cancellation can have large $k$. The measured check above is then the thing to repeat.

## 8. Try it yourself

Burgers with the bump-and-block data: pick $\delta$, the time and the projection. The left panel
compares $u^\delta$ with the exact solution. The right panel shows $U-U^\delta$ with the bound $\pm C_1\delta^2$,
which holds for the mass-preserving projection. The sliders work only in Jupyter or Colab.
"""),
        code(r'''
import ipywidgets as widgets
controls = dict(
    k=widgets.IntSlider(value=2, min=0, max=6, description="δ = 0.1/2^k"),
    T=widgets.FloatSlider(value=3.0, min=0.5, max=6.0, step=0.5, description="T"),
    projection=widgets.Dropdown(options=["mass-preserving", "nearest node"],
                                value="mass-preserving", description="projection"))
def show(k, T, projection):
    d = 0.1 / 2**k
    tr = (mass_projection(burgers, U0, [-1.0, 1.0, 1.5], d, 2 * A, 0.0, 1.0)
          if projection == "mass-preserving"
          else nearest_projection(burgers, u0, -1.0, 1.5, d, 0.0, 1.0)).run(T)
    x = np.sort(np.concatenate([np.linspace(-1.5, 1.6 + T, 20_001), front_positions(tr)]))
    u_ex, U_ex, _ = burgers_exact(x, T)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    a.plot(x, u_ex, color=C_EXACT, lw=1.5, label="exact")
    a.step(x, tr.sample(x), where="post", color=C_POLY, lw=1.2, label=r"$u^\delta$")
    a.set(xlabel="$x$", title=f"{len(tr.fronts)} fronts, {tr.event_count} events")
    a.legend(frameon=False)
    b.plot(x, U_ex - primitive(tr, x), color=C_ENV, lw=1.3)
    for s in (1, -1):
        b.axhline(s * (1 + T / 8) * d**2, color=C_FLUX, ls=":", lw=1.2)
    b.set(xlabel="$x$", title=r"$U - U^\delta$  (dotted: $\pm C_1\delta^2$)")
    plt.show()
display(widgets.HBox(list(controls.values()), layout=widgets.Layout(flex_flow="row wrap")),
        widgets.interactive_output(show, controls))
'''),
        md(r"""
## References

* C. M. Dafermos, Polygonal approximations of solutions of the initial value problem for a
  conservation law, *J. Math. Anal. Appl.* **38**(1) (1972) 33–41.
* S. N. Kružkov, First order quasilinear equations in several independent variables,
  *Math. USSR-Sb.* **10** (1970) 217–243.
* N. N. Kuznetsov, Accuracy of some approximate methods for computing the weak solutions of a
  first-order quasi-linear equation, *USSR Comput. Math. Math. Phys.* **16** (1976) 105–119.
* B. J. Lucier, A moving mesh numerical method for hyperbolic conservation laws,
  *Math. Comp.* **46** (1986) 59–69.
* F. Şabac, The optimal convergence rate of monotone finite difference methods for hyperbolic
  conservation laws, *SIAM J. Numer. Anal.* **34** (1997) 2306–2318.
* K. H. Karlsen, N. H. Risebro, A note on front tracking and the equivalence between viscosity
  solutions of Hamilton–Jacobi equations and entropy solutions of scalar conservation laws,
  *Nonlinear Anal.* **50** (2002) 455–469.
* S. Solem, Convergence rates of the front tracking method for conservation laws in the
  Wasserstein distances, *SIAM J. Numer. Anal.* **56** (2018) 3648–3666.
* S. S. Ghoshal, J. D. Towers, A convergence rate result for front tracking approximations of
  conservation laws with discontinuous flux, arXiv:2509.22952 (2025).
* J. R. Shewchuk, Adaptive precision floating-point arithmetic and fast robust geometric
  predicates, *Discrete Comput. Geom.* **18** (1997) 305–363. (Background on orientation tests.)
* H. Holden, N. H. Risebro, *Front Tracking for Hyperbolic Conservation Laws*, 2nd ed.,
  Springer (2015).

---
**Next:** *06 · Stability*: $L^1$ contraction, total variation, and how many fronts there can be.
"""),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "05_convergence.ipynb", "toc_visible": True},
    })
    return nb


def stability_notebook():
    cells = [
        md(r"""
# 06 · Stability: L¹ contraction, total variation, front count

Notebook 05 compared front tracking with the *exact* solution. This one compares front tracking
with **itself**. Take two step data $u_0^\delta$, $v_0^\delta$ on the same state grid, run both with the
same polygonal flux $f_\delta$, and ask what can happen to their difference, to each one's total
variation, and to the number of fronts.

Since $u^\delta$ and $v^\delta$ are exact entropy solutions of $u_t+f_\delta(u)_x=0$, every statement below
marked **exact** holds for them with no $\delta$-dependent error, up to floating point and the
cluster tolerance of notebook 04.

| claim | status |
|---|---|
| $\lVert u^\delta(t)-v^\delta(t)\rVert_{L^1}$ never increases ($L^1$ contraction) | exact (Kružkov) |
| its rate of decrease is an explicit sum over fronts | exact, derived in §2 |
| $u_0^\delta\le v_0^\delta\Rightarrow u^\delta\le v^\delta$ (comparison principle) | exact |
| $\mathrm{TV}(u^\delta(t))$ never increases, and changes only at events | exact, proof in §4 |
| number of fronts $N(t)\le \mathrm{TV}(u^\delta_0)/\delta_{\min}$ | exact, proof in §5 |
| number of events grows like $1/\delta$, even for nonconvex $f$ | observed |
| no stability in $L^\infty$ or $L^2$ | exact example, §6 |

Two things this notebook is **not** about. Contraction holds between two solutions with the **same**
$f_\delta$. Two different values of $\delta$ are two different equations, and their difference is the flux
term of notebook 05. And the front count in §5 is a measure of **cost**, not of stability. It lives here
because its bound comes from the total variation.

*Reference: S. N. Kružkov, First order quasilinear equations in several independent variables,
Math. USSR-Sb. 10 (1970) 217–243. Full references at the end.*
"""),
        code(SETUP),
        code(STYLE),
        md(r"""
## 1. $L^1$ contraction

**Theorem (Kružkov, 1970).** If $u$ and $v$ are entropy solutions of $u_t+g(u)_x=0$ with a Lipschitz
flux $g$, and $u_0-v_0\in L^1$, then for $0\le s\le t$
$$
\lVert u(t)-v(t)\rVert_{L^1}\le\lVert u(s)-v(s)\rVert_{L^1}.
$$
A polygonal flux $f_\delta$ is Lipschitz, so this applies to front tracking without change.

To measure it exactly we need no quadrature: two step functions are both constant between the
merged list of their fronts, so $\int|u^\delta-v^\delta|$ is a finite sum. The test below takes random
pairs of step data (same far-field states, so the distance is finite), runs both to $t=8$, and
records the largest *increase* of the distance between consecutive observation times.
"""),
        code(r'''
import math
from fronttrack.interactions import next_interaction

burgers = lambda u: 0.5 * u**2
sin3 = lambda u: np.sin(3 * u)
linear = lambda u: u
FLUXES = {"Burgers": burgers, "sin 3u": sin3, "linear f = u": linear}

def fronts_x(tr):
    return np.array([f.x for f in tr.fronts])

def l1_distance(a, b):
    """Exact L1 distance of two step solutions with equal far-field states."""
    p = np.unique(np.concatenate([fronts_x(a), fronts_x(b)]))
    if p.size < 2:
        return 0.0
    mid = 0.5 * (p[1:] + p[:-1])
    return math.fsum(np.abs(a.sample(mid) - b.sample(mid)) * np.diff(p))

def random_pair(flux, rng, jumps=20):
    """Two step data on the same jump positions; interior values independent, ends equal."""
    x = np.sort(rng.uniform(-3, 3, jumps))
    va, vb = rng.uniform(-1, 1, jumps + 1), rng.uniform(-1, 1, jumps + 1)
    vb[0], vb[-1] = va[0], va[-1]
    make = lambda v: FrontTracker.from_values(flux, x, v, record_history=False)
    return make(va), make(vb)

times = np.linspace(0, 8, 81)
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True, constrained_layout=True)
lines = [f"{'flux':14s} {'pairs':>5} {'largest increase':>17} {'final/initial (median)':>23}"]
for ax, (name, f) in zip(axes, FLUXES.items()):
    flux = DiscreteFlux(f, np.linspace(-1, 1, 41))
    worst, ratios = 0.0, []
    for k in range(40):
        a, b = random_pair(flux, rng)
        d = []
        for t in times:
            a.run(t); b.run(t)
            d.append(l1_distance(a, b))
        d = np.array(d)
        worst = max(worst, np.max(np.diff(d)))
        ratios.append(d[-1] / d[0])
        if k < 8:
            ax.plot(times, d, color=C_POLY, lw=1, alpha=0.7)
    ax.set(xlabel="$t$", title=name)
    lines.append(f"{name:14s} {40:5d} {worst:17.1e} {np.median(ratios):23.3f}")
axes[0].set_ylabel(r"$\|u^\delta(t)-v^\delta(t)\|_{L^1}$")
plt.show()
print("\n".join(lines))
'''),
        md(r"""
The largest increase is of order $10^{-15}$, which is roundoff: the distance never grows. For Burgers
and $\sin 3u$ it shrinks. For the linear flux it stays **exactly** constant, since every front moves at
speed $1$ and the whole picture just translates. Why nonlinear fluxes contract, and by how much,
has an exact answer.

## 2. Where the contraction comes from

Between two events, both solutions are step functions and all fronts move at constant speed, so
$\lVert u^\delta-v^\delta\rVert_{L^1}$ is differentiable in $t$. Consider a front of $u^\delta$ with states
$a\to b$ and speed $s=\frac{f_\delta(b)-f_\delta(a)}{b-a}$, at a point where $v^\delta$ has the constant value $c$.
(At all but finitely many times no front of $u^\delta$ sits exactly on a front of $v^\delta$.)

As the front moves, it sweeps a strip of width $s\,dt$ where $|u-v|$ changes from $|b-c|$ to $|a-c|$
(for $s>0$). Summing these rates over all fronts of both solutions gives $\frac{d}{dt}\lVert u-v\rVert_{L^1}$.
To see the sign, add the jumps of Kružkov's entropy flux $q(u,c)=\mathrm{sgn}(u-c)\,\big(f_\delta(u)-f_\delta(c)\big)$.
The function $x\mapsto q(u^\delta,v^\delta)$ is piecewise constant, jumps only at fronts, and vanishes at
$\pm\infty$, so its jumps sum to zero and adding them changes nothing. Each front then contributes
$$
E(a,b,c)=-s\,\big(|b-c|-|a-c|\big)+\big(q(b,c)-q(a,c)\big).
$$

* If $c$ is **not strictly between** $a$ and $b$, then $\mathrm{sgn}(u-c)$ is the same on both sides and
  $E=\pm\big(f_\delta(b)-f_\delta(a)-s(b-a)\big)=0$ by Rankine–Hugoniot.
* If $a<c<b$, a short computation using Rankine–Hugoniot gives
  $E=2\big(\ell(c)-f_\delta(c)\big)$, where $\ell$ is the chord from $(a,f_\delta(a))$ to $(b,f_\delta(b))$.
  The front is an edge of the lower convex envelope, so $f_\delta(c)\ge\ell(c)$ and $E\le0$. The case
  $a>c>b$ is the mirror image, with the upper concave envelope.

Together:
$$
\boxed{\;\frac{d}{dt}\lVert u^\delta-v^\delta\rVert_{L^1}
=-2\!\!\sum_{\substack{\text{fronts of } u^\delta \text{ or } v^\delta\\ \text{other solution's value } c \text{ strictly inside}}}\!\!\big|f_\delta(c)-\ell(c)\big|\;}
$$
This is Kružkov's entropy inequality with the constant $k$ chosen as the *other* solution's value, the
doubling of variables made concrete. It explains the pictures above.

* **Linear flux:** $f_\delta=\ell$ on every chord, so every term is zero and the distance is constant.
* **Nonlinear flux:** the distance drops only while a front of one solution crosses a region where the
  other solution's value lies strictly inside the front's jump, and by an amount set by how far $f_\delta$
  bulges away from that front's chord.

Since both solutions take node values, $c$ is a node, and $|f_\delta(c)-\ell(c)|$ is the gap of that node
from the chord (the same gap that `solve_riemann` tests, notebook 05 §7). Below, the formula is compared
with a one-sided finite difference of the exact distance, over a short step with no event in it.
"""),
        code(r'''
import copy

def dissipation_rate(a, b):
    """-2 * sum over fronts of |f_delta(c) - chord(c)|, c = other solution's value strictly inside."""
    g, fv = a.flux.u_grid, a.flux.f_values
    total = 0.0
    for u, v in ((a, b), (b, a)):
        for fr in u.fronts:
            c = v.sample(fr.x)
            lo, hi = sorted((g[fr.iL], g[fr.iR]))
            if lo < c < hi:
                k = int(np.searchsorted(g, c))              # c is a node value
                chord = fv[fr.iL] + fr.speed * (c - g[fr.iL])
                total -= 2 * abs(fv[k] - chord)
    return total

rng = np.random.default_rng(1)
lines = [f"{'flux':8s} {'checks':>6} {'largest |FD - formula| / (1 + |formula|)':>42}"]
for name in ("Burgers", "sin 3u"):
    flux = DiscreteFlux(FLUXES[name], np.linspace(-1, 1, 41))
    mism = []
    for trial in range(30):
        a, b = random_pair(flux, rng, jumps=12)
        for t in np.linspace(0.3, 4, 12):
            a.run(t); b.run(t)
            h = 1e-7 * min(next_interaction(a.fronts)[0], next_interaction(b.fronts)[0], 1.0)
            a2, b2 = copy.deepcopy(a), copy.deepcopy(b)
            a2.run(t + h); b2.run(t + h)
            if (a2.event_count, b2.event_count) != (a.event_count, b.event_count):
                continue                                    # an event inside the step: skip
            fd = (l1_distance(a2, b2) - l1_distance(a, b)) / h
            r = dissipation_rate(a, b)
            mism.append(abs(fd - r) / (1 + abs(r)))
    lines.append(f"{name:8s} {len(mism):6d} {max(mism):42.1e}")
print("\n".join(lines))
'''),
        md(r"""
The mismatch is at the level of the finite difference's own roundoff (a difference of two $O(1)$
numbers divided by a step of about $10^{-7}$), so the identity holds.

## 3. Comparison principle

**Corollary.** If $u_0^\delta\le v_0^\delta$ everywhere, then $u^\delta(t)\le v^\delta(t)$ for all $t$.

The same computation with $|\cdot|$ replaced by the positive part $(\cdot)^+$ shows that
$\int(u^\delta-v^\delta)^+$ is nonincreasing (Kružkov's proof gives this form too). It starts at $0$, so it
stays $0$. The test raises a random set of interior values of $u_0$ by a random nonnegative amount
and tracks $\int(u^\delta-v^\delta)^+\,dx$ exactly, over the merged fronts.

Why the integral and not $\min_x(v^\delta-u^\delta)$? Where the two solutions share a front (the same
fan, from the same jump), its two copies should sit at the same point, but they reach it through
different sequences of floating-point operations. They can end up one rounding error apart, and in
that sliver of width about $10^{-16}$ one solution has crossed the front and the other has not. A
pointwise test reports a violation of size $\delta$ there. The integral weighs it by its width, as the
theorem does.
"""),
        code(r'''
rng = np.random.default_rng(2)
lines = [f"{'flux':14s} {'pairs':>5} {'largest integral of (u - v)+':>29}"]
for name, f in FLUXES.items():
    flux = DiscreteFlux(f, np.linspace(-1, 1, 41))
    worst = 0.0
    for trial in range(30):
        x = np.sort(rng.uniform(-3, 3, 20))
        ua = rng.uniform(-1, 0.6, 21)
        ub = np.clip(ua + rng.uniform(0, 0.4, 21) * (rng.random(21) < 0.5), -1, 1)
        ub[0], ub[-1] = ua[0], ua[-1]
        a = FrontTracker.from_values(flux, x, ua, record_history=False)
        b = FrontTracker.from_values(flux, x, ub, record_history=False)
        for t in np.linspace(0, 8, 41):
            a.run(t); b.run(t)
            p = np.unique(np.concatenate([fronts_x(a), fronts_x(b)]))
            mid = 0.5 * (p[1:] + p[:-1])
            pos = math.fsum(np.clip(a.sample(mid) - b.sample(mid), 0, None) * np.diff(p))
            worst = max(worst, pos)
    lines.append(f"{name:14s} {30:5d} {worst:29.1e}")
print("\n".join(lines))
'''),
        md(r"""
$\int(u^\delta-v^\delta)^+$ stays at roundoff level: the ordering is never violated on any set of positive
length.

## 4. Total variation

**Claim.** $\mathrm{TV}(u^\delta(t))$ is constant between events and never increases at an event.

*Proof.* Between events every front keeps its states and only moves, so the list of jumps, and with it
the total variation, is unchanged. At an event a cluster of fronts meets at one point. Let
$u_\ell, u_1,\dots,u_m, u_r$ be the states from left to right. The incoming variation is
$|u_1-u_\ell|+\dots+|u_r-u_m|\ge|u_r-u_\ell|$. The outgoing fan is the Riemann solution from $u_\ell$ to
$u_r$, and its states are **monotone** (they are the vertices of one envelope, in order), so its variation is
exactly $|u_r-u_\ell|$. $\square$

TV drops strictly at an event exactly when the incoming states are not monotone. To watch this
event by event, the cell below subclasses `FrontTracker` so that it logs the total variation before and
after every event it resolves. The plot shows $\mathrm{TV}(t)$ as a step function, and the table counts
how many events decreased it.
"""),
        code(r'''
class RecordingTracker(FrontTracker):
    """FrontTracker that logs (time, TV before, TV after) at every event it resolves."""
    def _resolve_cluster(self, pair_index):
        before = self.total_variation()
        super()._resolve_cluster(pair_index)
        self.log.append((self.time, before, self.total_variation()))

def tv_history(flux, x, values, T):
    tr = RecordingTracker.from_values(flux, x, values, record_history=False)
    tr.log = []
    tv0 = tr.total_variation()
    tr.run(T)
    ts = np.array([0.0] + [e[0] for e in tr.log] + [T])
    tvs = np.array([tv0] + [e[2] for e in tr.log] + [tr.total_variation()])
    drops = sum(after < before - 1e-12 for _, before, after in tr.log)
    rises = max([after - before for _, before, after in tr.log], default=0.0)
    return ts, tvs, tr.event_count, drops, rises

rng = np.random.default_rng(3)
x, values = np.sort(rng.uniform(-3, 3, 30)), rng.uniform(-1, 1, 31)
fig, ax = plt.subplots(figsize=(6.5, 4))
lines = [f"{'flux':14s} {'TV(0)':>7} {'TV(8)':>7} {'events':>7} {'events that lowered TV':>23}"
         f" {'largest increase at an event':>29}"]
for (name, f), c in zip(FLUXES.items(), (C_POLY, C_ENV, C_FLUX)):
    ts, tvs, events, drops, rises = tv_history(DiscreteFlux(f, np.linspace(-1, 1, 41)),
                                               x, values, 8.0)
    ax.step(ts, tvs, where="post", color=c, label=name)
    lines.append(f"{name:14s} {tvs[0]:7.3f} {tvs[-1]:7.3f} {events:7d} {drops:23d}"
                 f" {max(0.0, rises):17.1e}")
ax.set(xlabel="$t$", ylabel=r"TV$(u^\delta(t))$", title="Total variation only steps down")
ax.legend(frameon=False)
plt.show()
print("\n".join(lines))
'''),
        md(r"""
TV never increases. For the linear flux nothing ever collides (all speeds are equal), so TV is
constant. For the nonlinear fluxes most events lower TV, and a minority leave it unchanged. Those are
collisions whose incoming states were already monotone, for instance two shocks in the same direction
merging into one shock with the same total jump. A shock absorbing one step of a rarefaction always
lowers TV, since the step goes the other way.

**A caveat about the data.** The claim is about $\mathrm{TV}(u^\delta(t))\le\mathrm{TV}(u_0^\delta)$, the
variation of the *projected* data. Nearest-node rounding can make $\mathrm{TV}(u_0^\delta)$ much larger than
$\mathrm{TV}(u_0)$. A small wiggle around the midpoint between two nodes becomes a full jump of size
$\delta$ at every crossing:
"""),
        code(r'''
flux = DiscreteFlux(burgers, np.linspace(-1, 1, 21))       # nodes ..., 0.0, 0.1, ...
xs = np.linspace(-1, 1, 401)
wiggle = 0.05 + 0.002 * np.sin(40 * xs)                    # stays within 0.048 .. 0.052
edges = 0.5 * (xs[1:] + xs[:-1])
tr = FrontTracker.from_values(flux, edges, wiggle, record_history=False)
print(f"TV of the sampled data:        {np.sum(np.abs(np.diff(wiggle))):.3f}")
print(f"TV after nearest-node rounding: {tr.total_variation():.3f}"
      f"   ({len(tr.fronts)} fronts from {len(edges)} jumps)")
'''),
        md(r"""
Here rounding multiplies the variation by about $25$. The cell averages of notebook 05 §2 never
increase TV, which is one more reason to prefer them.

## 5. How many fronts?

**Claim.** For any flux, $N(t)\le\mathrm{TV}(u^\delta_0)/\delta_{\min}$, where $\delta_{\min}$ is the smallest
spacing of the state grid.

*Proof.* Every front joins two **different** nodes, so its jump is at least $\delta_{\min}$. Adding up the jumps,
$N(t)\,\delta_{\min}\le\mathrm{TV}(u^\delta(t))\le\mathrm{TV}(u_0^\delta)$ by §4. $\square$

The number of fronts is therefore bounded for all time, by a constant of order $1/\delta$. The number of
*events* is a different matter. For strictly convex $f$, every binary collision removes a front
(notebook 04, §6), so $E(t)\le N(0)$. For nonconvex $f$ that argument fails, and finiteness of the event
count needs a separate proof (see Holden–Risebro, Ch. 2). Here is what happens in practice, for fixed
data and a refined state grid:
"""),
        code(r'''
rng = np.random.default_rng(3)
x, values = np.sort(rng.uniform(-3, 3, 40)), rng.uniform(-1, 1, 41)
node_counts = [21, 41, 81, 161, 321]
lines = [f"{'flux':8s} {'nodes':>6} {'N(0)':>6} {'events E(10)':>13} {'E/N(0)':>7} {'E ratio':>8}"
         f" {'max N(t) * d_min / TV(0)':>26}"]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for (name, f), c in zip((("Burgers", burgers), ("sin 3u", sin3), ("u³", lambda u: u**3)),
                        (C_POLY, C_ENV, C_FLUX)):
    E, prev = [], None
    for n in node_counts:
        flux = DiscreteFlux(f, np.linspace(-1, 1, n))
        tr = FrontTracker.from_values(flux, x, values, record_history=False)
        N0, tv0, Nmax = len(tr.fronts), tr.total_variation(), len(tr.fronts)
        for t in np.linspace(0, 10, 21)[1:]:
            tr.run(t)
            Nmax = max(Nmax, len(tr.fronts))
        E.append(tr.event_count)
        ratio = "" if prev is None else f"{E[-1] / prev:8.2f}"
        prev = E[-1]
        lines.append(f"{name:8s} {n:6d} {N0:6d} {E[-1]:13d} {E[-1] / N0:7.2f} {ratio:>8}"
                     f" {Nmax * (2 / (n - 1)) / tv0:26.3f}")
    ax.loglog(2 / (np.array(node_counts) - 1), E, "o-", color=c, ms=5, label=name)
d = 2 / (np.array(node_counts) - 1)
ax.loglog(d, 3 / d, ":", color=C_EXACT, lw=1, label=r"$\propto 1/\delta$")
ax.set(xlabel=r"$\delta$", ylabel="events up to t = 10", title="Event count vs state spacing")
ax.legend(frameon=False)
plt.show()
print("\n".join(lines))
'''),
        md(r"""
The last column is at most $1$, as the claim requires. The event count doubles each time $\delta$ is
halved, for the nonconvex fluxes as well as for Burgers, and $E/N(0)$ stays close to $1$ for Burgers and
near $1.2$ for the others. That linear growth is an observation for these data, not a theorem.

For the cost of a computation this means: about $1/\delta$ events, each found by the $O(N)$ scan of notebook
04, so about $1/\delta^2$ operations in all. A priority queue of collision times would bring this down to
about $(1/\delta)\log(1/\delta)$.

## 6. What is not stable

$L^1$ is special. The same solutions are **not** stable in $L^\infty$ or in $L^2$. Take Burgers with a single
shock $u_0=\mathbf 1_{(-1,0)}$ and the slightly higher $v_0=(1+\varepsilon)\,\mathbf 1_{(-1,0)}$. Their right edges
are shocks with speeds $\tfrac12$ and $\tfrac{1+\varepsilon}2$, so after time $t$ the two shocks are
$\varepsilon t/2$ apart, and in between the two solutions differ by about $1$. The grid below is chosen so
that $0$, $1$ and $1+\varepsilon$ are nodes and everything is exact. All three distances are computed
exactly from the fronts.
"""),
        code(r'''
def norms(a, b):
    """Exact L1, L2 and Linf distances of two step solutions (same far field)."""
    p = np.unique(np.concatenate([fronts_x(a), fronts_x(b)]))
    mid = 0.5 * (p[1:] + p[:-1])
    diff, w = np.abs(a.sample(mid) - b.sample(mid)), np.diff(p)
    return np.sum(diff * w), np.sqrt(np.sum(diff**2 * w)), diff.max()

def shock_pair(eps):
    """u0 = 1 on (-1,0), v0 = 1+eps on (-1,0); nodes 0, 1/2, 1, 1+eps make everything exact."""
    flux = DiscreteFlux(burgers, np.array([0.0, 0.5, 1.0, 1.0 + eps]))
    make = lambda top: FrontTracker.from_values(flux, [-1.0, 0.0], [0.0, top, 0.0],
                                                record_history=False)
    return make(1.0), make(1.0 + eps)

eps = 0.01
a, b = shock_pair(eps)
ts = np.linspace(0, 4, 401)
hist = []
for t in ts:
    a.run(t); b.run(t)
    hist.append(norms(a, b))
hist = np.array(hist)

eps_list = 10.0 ** -np.arange(1, 6)
final = []
for e in eps_list:
    a, b = shock_pair(e)
    a.run(4.0); b.run(4.0)
    final.append(norms(a, b))
final = np.array(final)

fig, (p, q) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for k, (lab, c) in enumerate(((r"$L^1$", C_POLY), (r"$L^2$", C_ENV), (r"$L^\infty$", C_FLUX))):
    p.plot(ts, hist[:, k], color=c, label=lab)
p.plot(ts, np.sqrt(np.minimum(eps * ts / 2, eps)), ":", color=C_EXACT, lw=1.2,
       label=r"$\sqrt{\min(\varepsilon t/2,\ \varepsilon)}$")
p.set(yscale="log", xlabel="$t$", ylabel="distance between the two solutions",
      title=f"Same data, ε = {eps}: three norms")
p.legend(frameon=False, loc="center right")
q.loglog(eps_list, eps_list, "o-", color=C_EXACT, ms=4, lw=1, label="all three norms at t = 0 (= ε)")
q.loglog(eps_list, final[:, 0], "s--", color=C_POLY, ms=5, label=r"$L^1$ at t = 4")
q.loglog(eps_list, final[:, 1], "o-", color=C_ENV, ms=5, label=r"$L^2$ at t = 4")
q.loglog(eps_list, final[:, 2], "^-", color=C_FLUX, ms=5, label=r"$L^\infty$ at t = 4")
q.set(xlabel=r"size of the perturbation $\varepsilon$", title="Before and after, as ε → 0")
q.legend(frameon=False, fontsize=9)
plt.show()

lines = [f"eps = {eps}", f"{'t':>4} {'L1':>9} {'L2':>9} {'Linf':>9}"]
for t in (0.0, 0.5, 1.0, 2.0, 4.0):
    i = int(np.argmin(np.abs(ts - t)))
    lines.append(f"{t:4g} " + " ".join(f"{v:9.4f}" for v in hist[i]))
lines += ["", f"{'eps':>8} {'L2 at t=4':>10} {'ratio L2(4)/L2(0)':>18}"]
for e, (l1, l2, li) in zip(eps_list, final):
    lines.append(f"{e:8.0e} {l2:10.4f} {l2 / e:18.1f}")
print("\n".join(lines))
'''),
        md(r"""
* **$L^1$:** the distance stays exactly $\varepsilon$. It does not shrink either, and §2 says why: at every front
  of one solution, the other solution's value is an endpoint of the jump, never strictly inside it.
* **$L^\infty$:** the distance jumps from $\varepsilon$ to about $1$ as soon as $t>0$, because between the two shocks
  one solution is $0$ and the other is about $1$.
* **$L^2$:** the distance grows like $\sqrt{\varepsilon t/2}$ while the shocks separate (dotted line, left). At
  $t=2$ the faster shock has absorbed the extra step $1\to1+\varepsilon$ of its rarefaction, both shocks then
  have the same speed, and the gap freezes at $\varepsilon$. So the $L^2$ distance ends at $\sqrt\varepsilon=0.1$, ten
  times its initial value.

The right panel repeats this for $\varepsilon=10^{-1},\dots,10^{-5}$. At $t=0$ all three distances equal $\varepsilon$.
At $t=4$ the $L^1$ distance still equals $\varepsilon$ (its markers sit on the black line), the $L^\infty$ distance
is about $1$ whatever $\varepsilon$ is, and the $L^2$ distance is $\sqrt\varepsilon$: slope $\tfrac12$ instead of $1$. The
amplification factor $L^2(4)/L^2(0)=\varepsilon^{-1/2}$ is **unbounded** as $\varepsilon\to0$, so no inequality
$\lVert u(t)-v(t)\rVert_{L^2}\le C\,\lVert u_0-v_0\rVert_{L^2}$ can hold, for any constant $C$. The data-to-solution
map is still continuous in $L^2$ here, but only Hölder with exponent $\tfrac12$.

Shocks are the reason: a small change of the data moves a jump, and in $L^p$ with $p>1$ a moved jump
costs far more than the change that moved it.

## 7. Try it yourself

Two random step data with the same far field, any flux. The left panel shows the $L^1$ distance together
with the rate formula of §2 integrated in time (a Riemann sum with step $0.005$; the rate is piecewise
constant, so the two curves should nearly coincide). The right panel shows both total variations. The sliders work only
in Jupyter or Colab.
"""),
        code(r'''
import ipywidgets as widgets
controls = dict(
    name=widgets.Dropdown(options=list(FLUXES) + ["u³", "traffic u(1-u)"], value="sin 3u",
                          description="flux"),
    nodes=widgets.IntSlider(value=41, min=5, max=161, step=4, description="nodes"),
    jumps=widgets.IntSlider(value=15, min=2, max=40, description="jumps"),
    seed=widgets.IntSlider(value=0, min=0, max=20, description="seed"))
more = {"u³": lambda u: u**3, "traffic u(1-u)": lambda u: u * (1 - u)}
def show(name, nodes, jumps, seed):
    flux = DiscreteFlux({**FLUXES, **more}[name], np.linspace(-1, 1, nodes))
    a, b = random_pair(flux, np.random.default_rng(seed), jumps)
    ts = np.linspace(0, 8, 1601)   # fine steps: the rate is piecewise constant
    d, tva, tvb, integ = [], [], [], [0.0]
    for k, t in enumerate(ts):
        if k:
            integ.append(integ[-1] + dissipation_rate(a, b) * (t - ts[k - 1]))
        a.run(t); b.run(t)
        d.append(l1_distance(a, b)); tva.append(a.total_variation()); tvb.append(b.total_variation())
    fig, (p, q) = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    p.plot(ts, d, color=C_POLY, label=r"$\|u^\delta-v^\delta\|_{L^1}$")
    p.plot(ts, d[0] + np.array(integ), "--", color=C_ENV, lw=1.2,
           label="initial + integrated rate of §2 (step 0.005)")
    p.set(xlabel="$t$", title="L¹ distance")
    p.legend(frameon=False, fontsize=8)
    q.plot(ts, tva, color=C_POLY, label="TV(u)"); q.plot(ts, tvb, color=C_ENV, label="TV(v)")
    q.set(xlabel="$t$", title="total variation")
    q.legend(frameon=False)
    plt.show()
display(widgets.HBox(list(controls.values()), layout=widgets.Layout(flex_flow="row wrap")),
        widgets.interactive_output(show, controls))
'''),
        md(r"""
## References

* S. N. Kružkov, First order quasilinear equations in several independent variables,
  *Math. USSR-Sb.* **10** (1970) 217–243.
* C. M. Dafermos, Polygonal approximations of solutions of the initial value problem for a
  conservation law, *J. Math. Anal. Appl.* **38**(1) (1972) 33–41.
* H. Holden, N. H. Risebro, *Front Tracking for Hyperbolic Conservation Laws*, 2nd ed.,
  Springer (2015), Ch. 2.

---
This is the last notebook of the series. **03** built the Riemann solver, **04** the event loop, **05**
measured how close the result is to the true solution, and **06** how the method behaves as a
dynamical system in its own right.
"""),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "06_stability.ipynb", "toc_visible": True},
    })
    return nb


NOTEBOOKS = {"03_riemann_problem.ipynb": riemann_notebook,
             "04_interactions.ipynb": interactions_notebook,
             "05_convergence.ipynb": convergence_notebook,
             "06_stability.ipynb": stability_notebook}


WIDGET_MIME = "application/vnd.jupyter.widget-view+json"
PREVIEW_NOTE = ("Static preview at the default settings. Run this cell in Jupyter "
                "or Colab to get the sliders.\n")


def is_widget_cell(cell):
    """Widget cells follow one convention: they define `controls` and `show`."""
    return cell.cell_type == "code" and "interactive_output(show, controls)" in cell.source


def add_preview_probes(nb):
    """After each widget cell, add a temporary cell that draws the default view."""
    cells = []
    for cell in nb.cells:
        cells.append(cell)
        if is_widget_cell(cell):
            probe = code("show(**{key: w.value for key, w in controls.items()})")
            probe.metadata["build_probe"] = True
            cells.append(probe)
    nb.cells = cells


def replace_widgets_with_previews(nb):
    """Swap each saved widget pointer for the probe's static figure.

    A saved widget output is only a model id for a live kernel object, so viewers
    without a kernel (GitHub, VS Code, nbviewer) show "Could not render". The
    widget state in the metadata also uses random ids, which churned every rebuild.
    """
    cells = []
    for cell in nb.cells:
        if cell.metadata.get("build_probe"):
            widget_cell = cells[-1]
            widget_cell.outputs = [nbf.v4.new_output("stream", name="stdout", text=PREVIEW_NOTE)]
            widget_cell.outputs += [o for o in cell.outputs if o.output_type == "display_data"
                                    and WIDGET_MIME not in o.get("data", {})]
            continue
        cells.append(cell)
    nb.cells = cells
    nb.metadata.pop("widgets", None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="run and save outputs")
    args = parser.parse_args()
    for name, build in NOTEBOOKS.items():
        nb = build()
        path = ROOT / "notebooks" / name
        if args.execute:
            from nbclient import NotebookClient
            add_preview_probes(nb)
            NotebookClient(nb, timeout=600, kernel_name="python3", record_timing=False,
                           resources={"metadata": {"path": str(path.parent)}}).execute()
            replace_widgets_with_previews(nb)
        # nbformat gives each new cell a random id, so every rebuild would change
        # every id and bury the real change in a noisy diff. Number them instead.
        for k, cell in enumerate(nb.cells):
            cell.id = f"{name[:2]}-{k:02d}"
        nbf.write(nb, path)
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
