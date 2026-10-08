"""Draw the forward (alpha), backward (beta) and posterior (gamma) figures for the
"Breaking down the CTC Loss" post as SVG.

The cells, zeros and arrows are computed from the actual recursions (CMU / Raj
convention: beta excludes y at its own timestep), so the figures always agree
with the equations in the post.

Usage: python3 _scripts/ctc_figures.py   (writes images/ctc_loss/{alpha,beta,gamma}_prob.svg)
"""
from pathlib import Path

LABEL = "door"
TOKENS = ["ε"] + [c for ch in LABEL for c in (ch, "ε")]   # ε d ε o ε o ε r ε
S, T = len(TOKENS), 10
SHOWN = [0, 1, 2, 3, None, T - 2, T - 1]                   # None = "• • •"
OUT = Path(__file__).resolve().parent.parent / "images" / "ctc_loss"


# ---------------------------------------------------------------- recursions
def skip_from(s):          # alpha: may jump from s-2 to s
    return s >= 2 and TOKENS[s] != "ε" and TOKENS[s] != TOKENS[s - 2]


def skip_to(s):            # beta: may jump from s to s+2
    return s + 2 < S and TOKENS[s] != "ε" and TOKENS[s] != TOKENS[s + 2]


def preds(s):
    return [r for r in (s, s - 1, s - 2) if r >= 0 and (r != s - 2 or skip_from(s))]


def succs(s):
    return [r for r in (s, s + 1, s + 2) if r < S and (r != s + 2 or skip_to(s))]


alpha = [[False] * T for _ in range(S)]       # non-zero pattern (any y > 0)
alpha[0][0] = alpha[1][0] = True
for t in range(1, T):
    for s in range(S):
        alpha[s][t] = any(alpha[r][t - 1] for r in preds(s))

beta = [[False] * T for _ in range(S)]
beta[S - 1][T - 1] = beta[S - 2][T - 1] = True
for t in range(T - 2, -1, -1):
    for s in range(S):
        beta[s][t] = any(beta[r][t + 1] for r in succs(s))

gamma = [[alpha[s][t] and beta[s][t] for t in range(T)] for s in range(S)]

# ---------------------------------------------------------------- drawing
CELL, GAP_X, X0, Y0 = 84, 96, 130, 220
COL_X = {}
x = X0
for c in SHOWN:
    if c is None:
        x += 104
        continue
    COL_X[c] = x
    x += CELL + GAP_X
WIDTH = x - GAP_X + 70
HEIGHT = Y0 + S * CELL + 170
MATH = "'Latin Modern Math','STIX Two Math','Cambria Math','Times New Roman',serif"
SANS = "'Inter','Helvetica Neue',Arial,sans-serif"
ARROW = "#9a3412"
GRID = "#2f45d8"


def sub(sym, idx, size=26, italic=True):
    """Symbol with a parenthesised subscript, e.g. sub('α', '0,1')."""
    style = ' font-style="italic"' if italic else ""
    return (f'<tspan{style}>{sym}</tspan>'
            f'<tspan dy="7" font-size="{size * 0.62:.0f}">({idx})</tspan><tspan dy="-7"></tspan>')


def math_text(x, y, parts, size=26, anchor="middle"):
    return (f'<text x="{x}" y="{y}" font-family="{MATH}" font-size="{size}" '
            f'text-anchor="{anchor}" fill="#111">{"".join(parts)}</text>')


def cell_center(s, t):
    return COL_X[t] + CELL / 2, Y0 + s * CELL + CELL / 2


# Each arrow gets its own exit/entry point by row offset (0 = same row, 1 = next row,
# 2 = skip), so arrows into or out of the same cell never share an endpoint.
EXIT = {0: 0.50, 1: 0.70, 2: 0.88}
ENTRY = {0: 0.50, 1: 0.30, 2: 0.12}


def arrow(s1, t1, s2, t2):
    d = s2 - s1
    x1, y1 = COL_X[t1] + CELL, Y0 + s1 * CELL + CELL * EXIT[d]
    x2, y2 = COL_X[t2] - 3, Y0 + s2 * CELL + CELL * ENTRY[d]
    return (f'<line x1="{x1}" y1="{y1:.1f}" x2="{x2}" y2="{y2:.1f}" stroke="{ARROW}" '
            f'stroke-width="1.8" marker-end="url(#head)"/>')


def callout(text_parts, tx, ty, s, t, anchor="middle", size=27, target="right", lx=None):
    """Formula label at (tx, ty) with a leader line to the top or right edge of cell (s, t).
    lx: x where the leader leaves the label (defaults to tx)."""
    cx, cy = cell_center(s, t)
    ex, ey = (cx + 22, Y0 + s * CELL) if target == "top" else (COL_X[t] + CELL, cy)
    lx = tx if lx is None else lx
    ly = ty + 20 if ty < ey else ty - 28
    return (f'<line x1="{lx}" y1="{ly}" x2="{ex}" y2="{ey}" stroke="#555" stroke-width="1.2"/>'
            + math_text(tx, ty, text_parts, size, anchor))


def callout_below(text_parts, ty, s, t, lane=0, size=27):
    """Label below the grid, right-aligned to the figure edge, with a leader that runs up
    the outside of the last column and into the right edge of cell (s, t)."""
    cx, cy = cell_center(s, t)
    rx = COL_X[t] + CELL + 14 + 14 * lane
    tx = rx - 10
    path = f'M{rx},{ty - 9} V{cy} H{COL_X[t] + CELL}'
    return (f'<path d="{path}" fill="none" stroke="#555" stroke-width="1.2"/>'
            + math_text(tx, ty, text_parts, size, "end"))


def figure(name, symbol, nonzero, value, arrows, callouts, extra=""):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
           f'width="{WIDTH}" height="{HEIGHT}">',
           f'<defs><marker id="head" markerWidth="9" markerHeight="7" refX="8" refY="3.5" '
           f'orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L9,3.5 L0,7 z" fill="{ARROW}"/></marker></defs>',
           f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#fff"/>',
           math_text(62, Y0 - 34, [sub(symbol, "s,t", 36)], 36)]
    for c in SHOWN:
        if c is None:
            dx = COL_X[3] + CELL + (COL_X[T - 2] - COL_X[3] - CELL) / 2
            for k in (-30, 0, 30):
                out.append(f'<circle cx="{dx + k}" cy="{Y0 + S * CELL / 2}" r="5" fill="{GRID}"/>')
            continue
        head = {T - 2: "T−2", T - 1: "T−1"}.get(c, str(c))
        out.append(f'<text x="{COL_X[c] + CELL / 2}" y="{Y0 - 14}" font-family="{SANS}" '
                   f'font-size="27" text-anchor="middle" fill="#111">{head}</text>')
    for s, tok in enumerate(TOKENS):
        y = Y0 + s * CELL + CELL / 2 + 10
        out.append(f'<text x="40" y="{y}" font-family="{MATH}" font-size="38" '
                   f'text-anchor="middle" fill="#111">{tok}</text>')
        out.append(f'<text x="96" y="{y - 2}" font-family="{SANS}" font-size="24" '
                   f'text-anchor="middle" fill="#111">{s}</text>')
        for t in COL_X:
            x, yy = COL_X[t], Y0 + s * CELL
            out.append(f'<rect x="{x}" y="{yy}" width="{CELL}" height="{CELL}" fill="#fff" '
                       f'stroke="{GRID}" stroke-width="2"/>')
            label = value(s, t) if nonzero[s][t] else None
            if label is None:
                out.append(f'<text x="{x + CELL / 2}" y="{yy + CELL / 2 + 10}" font-family="{SANS}" '
                           f'font-size="29" text-anchor="middle" fill="#111">0</text>')
            else:
                out.append(math_text(x + CELL / 2, yy + CELL / 2 + 9, label, 29))
    out += arrows + callouts + [extra, "</svg>"]
    (OUT / f"{name}.svg").write_text("\n".join(out))
    print("wrote", OUT / f"{name}.svg")


def forward_arrows(pattern):
    res = []
    pairs = [(0, 1), (1, 2), (2, 3), (T - 2, T - 1)]
    for t1, t2 in pairs:
        for s in range(S):
            if not pattern[s][t2]:
                continue
            for r in preds(s):
                if pattern[r][t1]:
                    res.append(arrow(r, t1, s, t2))
    return res


def backward_arrows(pattern):
    res = []
    for t1, t2 in [(0, 1), (1, 2), (2, 3), (T - 2, T - 1)]:
        for s in range(S):
            if not pattern[s][t1]:
                continue
            for r in succs(s):
                if pattern[r][t2]:
                    res.append(arrow(s, t1, r, t2))
    return res


A = lambda s, t: [sub("α", f"{s},{t}")]
B = lambda s, t: [sub("β", f"{s},{t}")]
G = lambda s, t: [sub("γ", f"{s},{t}")]
Y = lambda s, t: sub("y", f"{s},{t}")
DOT = '<tspan> · </tspan>'
PLUS = '<tspan> + </tspan>'
band, bottom = Y0 - 110, Y0 + S * CELL + 70
right = COL_X[T - 1] + CELL + 40

# ---- alpha
figure("alpha_prob", "α", alpha, A, forward_arrows(alpha), [
    callout([sub("α", "0,0"), '<tspan> = </tspan>', Y(0, 0), '<tspan>,  </tspan>',
             sub("α", "1,0"), '<tspan> = </tspan>', Y(1, 0)],
            COL_X[0] - 30, band - 40, 0, 0, anchor="start", target="top", lx=COL_X[0] + CELL / 2 + 22),
    callout([sub("α", "0,0"), DOT, Y(0, 1)], COL_X[1] + CELL / 2 + 40, band + 10, 0, 1, target="top",
            lx=COL_X[1] + CELL / 2 + 22),
    callout(["<tspan>(</tspan>", sub("α", "0,2"), PLUS, sub("α", "1,2"), "<tspan>)</tspan>", DOT, Y(1, 3)],
            COL_X[3] + CELL + 30, band, 1, 3, anchor="start", lx=COL_X[3] + CELL + 40),
    callout_below(["<tspan>(</tspan>", sub("α", "5,8"), PLUS, sub("α", "6,8"), PLUS, sub("α", "7,8"),
                   "<tspan>)</tspan>", DOT, Y(7, 9)], bottom + 52, 7, T - 1, lane=1),
    callout_below(["<tspan>(</tspan>", sub("α", "7,8"), PLUS, sub("α", "8,8"), "<tspan>)</tspan>", DOT,
                   Y(8, 9)], bottom + 4, 8, T - 1, lane=0),
])

# ---- beta (CMU convention: beta(T-1) = 1 for the last two states)
figure("beta_prob", "β", beta,
       lambda s, t: ['<tspan>1</tspan>'] if t == T - 1 else B(s, t),
       backward_arrows(beta), [
    callout([sub("β", "0,1"), DOT, Y(0, 1), PLUS, sub("β", "1,1"), DOT, Y(1, 1)],
            COL_X[0] - 20, band - 40, 0, 0, anchor="start", target="top", lx=COL_X[0] + CELL / 2 + 22),
    callout([sub("β", "1,4"), DOT, Y(1, 4), PLUS, sub("β", "2,4"), DOT, Y(2, 4), PLUS,
             sub("β", "3,4"), DOT, Y(3, 4)], COL_X[3] + CELL + 30, band, 1, 3,
            anchor="start", lx=COL_X[3] + CELL + 40),
    callout([sub("β", "3,4"), DOT, Y(3, 4), PLUS, sub("β", "4,4"), DOT, Y(4, 4)],
            COL_X[3] + CELL + 30, bottom, 3, 3, anchor="start", lx=COL_X[3] + CELL + 50),
])

# ---- gamma = alpha * beta; the last column sums to P(seq | x)
bx = COL_X[T - 1] + CELL + 10
y7, y9 = Y0 + 7 * CELL + 6, Y0 + 9 * CELL - 6
bracket = (f'<path d="M{bx},{y7} h14 V{y9} h-14" fill="none" stroke="#444" stroke-width="1.5"/>'
           f'<path d="M{bx + 14},{(y7 + y9) / 2} h16 V{bottom + 4 - 9}" fill="none" stroke="#555" stroke-width="1.2"/>'
           + math_text(bx + 20, bottom + 4,
                       [sub("γ", "7,9"), PLUS, sub("γ", "8,9"), '<tspan> = </tspan>',
                        '<tspan font-style="italic">P</tspan><tspan>(seq | </tspan>'
                        '<tspan font-style="italic">x</tspan><tspan>)</tspan>'], 27, "end"))
figure("gamma_prob", "γ", gamma, G, forward_arrows(gamma), [
    callout([sub("α", "0,0"), DOT, sub("β", "0,0")], COL_X[0] + CELL / 2, band - 40, 0, 0, target="top",
            lx=COL_X[0] + CELL / 2 + 22),
], bracket)
