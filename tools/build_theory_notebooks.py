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


NOTEBOOKS = {"03_riemann_problem.ipynb": riemann_notebook,
             "04_interactions.ipynb": interactions_notebook}


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
