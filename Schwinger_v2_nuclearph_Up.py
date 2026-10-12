#!/usr/bin/env python3

"""Schwinger V2: varijacioni kanali celih prelaza, bez proseka brojeva."""

import argparse
import csv
from pathlib import Path
import numpy as np

DEFAULTS = (
    "/data/loto7_4698_k80.csv",
    "/data/loto7_4698_k80_loto_2971.csv",
    "/data/loto7_4698_k80_loto_plus_1727.csv",
)

N, K, WINDOW, STARTS = 39, 7, 96, 6
RANKS = (2, 4, 8)
ALPHAS = (.1, 1., 10.)


def load(path):
    rows = []

    with Path(path).open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not v.strip() for v in row):
                continue

            try:
                values = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem cele brojeve."
                ) from error

            if (
                len(values) != K
                or values != sorted(values)
                or len(set(values)) != K
                or not all(1 <= v <= N for v in values)
            ):
                raise ValueError(
                    f"Red {line}: neispravna sortirana "
                    "7/39 kombinacija."
                )

            rows.append(values)

    return np.asarray(rows, dtype=int).reshape(-1, K)


def gaps(rows):
    rows = np.atleast_2d(rows)

    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), 40)
    ))

    return np.diff(edges, axis=1)


def incoming(g, previous):
    a = g/5.
    b = previous/5.

    return np.column_stack((
        np.ones(len(a)), a, b, a*b
    ))


def distances(a, b):
    result = np.zeros((len(a), len(b)))

    for j in range(a.shape[1]):
        result += np.abs(
            a[:, j, None]-b[None, :, j]
        )

    return result


def prepare(history):
    if len(history) < WINDOW+2:
        raise ValueError(
            f"Potrebno je najmanje "
            f"{WINDOW+2} prethodnih redova."
        )

    history = history[-(WINDOW+2):]
    g = gaps(history)
    changes = np.diff(g, axis=0)

    x = incoming(
        g[1:-1], changes[:-1]
    )
    query = incoming(
        g[-1:], changes[-1:]
    )[0]

    scale = np.maximum(
        np.sqrt(np.mean(x*x, axis=0)),
        1e-8
    )
    x, query = x/scale, query/scale

    # Svaki razlicit CELOKUPAN prelaz je zaseban kanal.
    # Cilj je jedinicni vektor kanala.
    # Ne predvidjaju se prosecni brojevi.
    channels, labels = np.unique(
        changes[1:],
        axis=0,
        return_inverse=True
    )
    targets = np.eye(len(channels))[labels]

    separation = distances(
        channels/5., channels/5.
    )
    np.fill_diagonal(separation, np.inf)

    closest = separation.min(axis=1)
    closest = closest[
        np.isfinite(closest)
        & (closest > 1e-12)
    ]

    bandwidth = (
        float(np.median(closest))
        if len(closest) else 1.
    )

    return (
        x, targets, query,
        channels/5., bandwidth,
        g[-1], history[2:]
    )


def variational_operator(prepared, rank, alpha):
    x, targets = prepared[:2]

    gram = (
        x.T @ x
        + alpha*np.eye(x.shape[1])
    )
    factor = np.linalg.cholesky(gram)

    transformed = np.linalg.solve(
        factor, x.T @ targets
    )

    left, singular, right = np.linalg.svd(
        transformed,
        full_matrices=False
    )

    retained = min(rank, len(singular))
    reduced = (
        left[:, :retained]*singular[:retained]
    ) @ right[:retained]

    operator = np.linalg.solve(
        factor.T, reduced
    )

    return operator, singular


def objective(prepared, operator, alpha):
    x, targets = prepared[:2]
    residual = targets-x @ operator

    return .5*float(
        np.sum(residual*residual)
        + alpha*np.sum(operator*operator)
    )


def fit(prepared, rank, alpha):
    operator, singular = variational_operator(
        prepared, rank, alpha
    )

    amplitude = prepared[2] @ operator
    norm = np.linalg.norm(amplitude)

    if not np.isfinite(norm) or norm <= 1e-12:
        raise ArithmeticError(
            "Neispravna amplituda kanala."
        )

    return (
        prepared[3],
        prepared[4],
        prepared[5],
        amplitude/norm,
        singular
    )


def scores(rows, model):
    channels, bandwidth, reference, amplitude = model[:4]

    transitions = (
        gaps(rows)-reference
    )/5.

    overlap = np.exp(
        -distances(transitions, channels)/bandwidth
    )

    # Numericka stabilnost kod udaljenih prelaza.
    overlap /= np.maximum(
        overlap.max(axis=1, keepdims=True),
        1e-300
    )
    overlap /= np.maximum(
        np.linalg.norm(
            overlap, axis=1, keepdims=True
        ),
        1e-300
    )

    result = np.abs(overlap @ amplitude)**2

    if not np.all(np.isfinite(result)):
        raise ArithmeticError(
            "Neispravan skor kanala."
        )

    return result


def replacements(row):
    unused = np.setdiff1d(np.arange(1, 40), row)
    result = np.tile(row, (K*len(unused), 1))

    for j in range(K):
        result[j*len(unused):(j+1)*len(unused), j] = unused

    return np.sort(result, axis=1)


def search(history, model):
    # Polazista su stvarni redovi.
    # Pretraga nastavlja zamene do lokalnog maksimuma.
    seeds = np.unique(history, axis=0)
    order = np.argsort(
        -scores(seeds, model),
        kind="stable"
    )
    current = seeds[order[:STARTS]]
    iterations = 0

    while True:
        iterations += 1

        options = np.vstack([
            np.vstack((row, replacements(row)))
            for row in current
        ])
        values = scores(
            options, model
        ).reshape(len(current), 225)

        chosen = np.argmax(values, axis=1)

        improved = (
            values[np.arange(len(current)), chosen]
            > values[:, 0]+1e-10
        )

        if not np.any(improved):
            best = int(np.argmax(values[:, 0]))

            return (
                tuple(map(int, current[best])),
                float(values[best, 0]),
                iterations
            )

        blocks = options.reshape(
            len(current), 225, K
        )
        current[improved] = blocks[
            np.arange(len(current))[improved],
            chosen[improved]
        ]


def predict(prepared, rank, alpha):
    return search(
        prepared[6],
        fit(prepared, rank, alpha)
    )


def hits(prediction, actual):
    return len(
        set(map(int, prediction))
        & set(map(int, actual))
    )


def summarize(label, values):
    print(
        f"{label}: prosek={np.mean(values):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(values, minlength=8).tolist()}",
        flush=True
    )


def run_file(path, validation_count=48, test_count=64):
    rows = load(path)
    split = len(rows)-test_count
    begin = split-validation_count

    if (
        validation_count < 16
        or test_count < 16
        or begin < WINDOW+2
    ):
        raise ValueError(
            "Premalo istorije ili manje od "
            "16 validacionih/test kola."
        )

    settings = tuple(
        (r, a) for r in RANKS for a in ALPHAS
    )
    validation = [[] for _ in settings]

    print(
        f"\nUlaz: {Path(path).resolve()}; "
        f"redova={len(rows)}",
        flush=True
    )

    for t in range(begin, split):
        p = prepare(rows[:t])

        for j, (rank, alpha) in enumerate(settings):
            combo, _, _ = predict(
                p, rank, alpha
            )
            validation[j].append(
                hits(combo, rows[t])
            )

        if (t-begin+1) % 16 == 0:
            print(
                f"Validacija: "
                f"{t-begin+1}/{validation_count}",
                flush=True
            )

    means = [np.mean(v) for v in validation]
    chosen = int(np.argmax(means))
    rank, alpha = settings[chosen]

    print(
        f"Validacija: rang={rank}; "
        f"alpha={alpha:g}; "
        f"prosek={means[chosen]:.4f}",
        flush=True
    )

    actual_scores, previous_scores = [], []

    # Rang i alpha ostaju fiksni tokom testa.
    # Svako kolo koristi samo prethodne redove.
    for t in range(split, len(rows)):
        combo, _, _ = predict(
            prepare(rows[:t]),
            rank,
            alpha
        )

        actual_scores.append(
            hits(combo, rows[t])
        )
        previous_scores.append(
            hits(rows[t-1], rows[t])
        )

        if (t-split+1) % 16 == 0:
            print(
                f"Test: {t-split+1}/{test_count}",
                flush=True
            )

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize(
        "SCHWINGER V2 NUCLEAR PHYSICS",
        actual_scores
    )
    summarize(
        "Prethodna kombinacija",
        previous_scores
    )

    p = prepare(rows)
    model = fit(p, rank, alpha)
    combo, score, iterations = search(p[6], model)

    repeated = any(
        np.array_equal(combo, row)
        for row in rows
    )

    print(
        f"Poslednji red: "
        f"{','.join(map(str, rows[-1]))}"
    )
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"Skor={score:.10f}; "
        f"iteracija={iterations}; "
        f"vec postoji u istoriji={repeated}"
    )
    print(
        "Singularne vrednosti:",
        ",".join(
            f"{v:.6f}" for v in model[4][:rank]
        )
    )
    print(
        "Proveren lokalni maksimum svih 224 zamena; "
        "globalni maksimum nije potvrden."
    )
    print(
        "Kanali su strukturna analogija; "
        "skor nije verovatnoca dobitka."
    )


def self_test():
    base = np.array([
        [1,2,3,4,5,6,7],
        [4,12,18,28,35,36,37],
        [33,34,35,36,37,38,39],
        [2,7,12,18,24,30,39]
    ])
    history = np.tile(base, (30, 1))
    p = prepare(history)
    x, targets = p[:2]

    assert np.allclose(
        targets.sum(axis=1), 1.
    )

    for rank in RANKS:
        op, singular = variational_operator(
            p, rank, 1.
        )
        full, _ = variational_operator(
            p,
            min(x.shape[1], targets.shape[1]),
            1.
        )

        assert (
            np.linalg.matrix_rank(op, tol=1e-9)
            <= rank
        )
        assert np.isclose(
            objective(p, op, 1.)
            - objective(p, full, 1.),
            .5*np.sum(singular[rank:]**2),
            atol=1e-8
        )

        gradient = (
            x.T @ (x @ op-targets)+op
        )

        assert np.allclose(
            op.T @ gradient, 0., atol=1e-8
        )
        assert np.allclose(
            gradient @ op.T, 0., atol=1e-8
        )

        result = predict(p, rank, 1.)
        assert result == predict(
            prepare(history), rank, 1.
        )

        model = fit(p, rank, 1.)
        assert (
            scores(
                replacements(result[0]), model
            ).max()
            <= result[1]+1e-9
        )

    altered = history.copy()
    altered[-1] = base[0]

    assert (
        predict(prepare(history[:-1]), 4, 1.)
        == predict(prepare(altered[:-1]), 4, 1.)
    )

    print(
        "Self-test OK: kanali, varijacioni minimum, "
        "rang, hronologija i pretraga."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*")
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    for path in args.csv or DEFAULTS:
        run_file(path, args.validation, args.test)


if __name__ == "__main__":
    main()



"""
Ulaz: /data/loto7_4698_k80.csv; redova=4698
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: rang=2; alpha=0.1; prosek=1.3542
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 4635..4698
SCHWINGER V2 NUCLEAR PHYSICS: prosek=1.2031; pogodaka 0..7=[11, 34, 14, 5, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 5,x,11,y,34,z,36
Skor=0.8454760830; iteracija=14; vec postoji u istoriji=False
Singularne vrednosti: 0.999923,0.999829
Proveren lokalni maksimum svih 224 zamena; globalni maksimum nije potvrden.
Kanali su strukturna analogija; skor nije verovatnoca dobitka.



Ulaz: /data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: rang=2; alpha=0.1; prosek=1.1458
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 2908..2971
SCHWINGER V2 NUCLEAR PHYSICS: prosek=1.2188; pogodaka 0..7=[16, 26, 14, 8, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 13,x,25,y,27,z,39
Skor=0.7477856078; iteracija=10; vec postoji u istoriji=False
Singularne vrednosti: 0.999924,0.999845
Proveren lokalni maksimum svih 224 zamena; globalni maksimum nije potvrden.
Kanali su strukturna analogija; skor nije verovatnoca dobitka.



Ulaz: /data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: rang=2; alpha=0.1; prosek=1.3958
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 1664..1727
SCHWINGER V2 NUCLEAR PHYSICS: prosek=1.3750; pogodaka 0..7=[7, 35, 13, 9, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 1,x,19,y,33,z,37
Skor=0.8304850480; iteracija=7; vec postoji u istoriji=False
Singularne vrednosti: 0.999921,0.999814
Proveren lokalni maksimum svih 224 zamena; globalni maksimum nije potvrden.
Kanali su strukturna analogija; skor nije verovatnoca dobitka.
"""



"""
Varijacioni operator sada uči kanale celih prelaza, umesto prosečnih položaja brojeva. 
Završni predlozi su različiti i nijedan već ne postoji u svom CSV-u.

Fajl	                Sledeća kombinacija	    Prosek pogodaka na 64 kola	Prethodna kombinacija
Objedinjeni, red 4699	5,x,11,y,34,z,36	    1,2031	                    1,2188
Loto, red 2972	        13,x,25,y,27,z,39	    1,2188	                    1,2969
Loto Plus, red 1728	    1,x,19,y,33,z,37	    1,3750	                    1,2969

Prednost na ovom testu postoji samo za Loto Plus. Provereni su varijacioni minimum, rang operatora, hronologija, ponovljivost i svih 224 zamena završnih predloga. Globalni maksimum pretrage nije potvrđen.
"""

 

"""
Julian Seymour Schwinger  
1-quantum electrodynamics.   2-nuclear physics.     3-particle physics.     
4-atomic physics       5-quantum field theory.        6-Schwinger effect
"""
