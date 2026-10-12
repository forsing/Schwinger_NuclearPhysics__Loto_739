#!/usr/bin/env python3

"""SCHWINGER V2 — NUCLEAR PHYSICS: variational transition channels."""

import argparse
import csv
import itertools
from pathlib import Path
import numpy as np



DEFAULT = "/data/loto7_4698_k80.csv"
# DEFAULT = "/data/loto7_4698_k80_loto_2971.csv"
# DEFAULT = "/data/loto7_4698_k80_loto_plus_1727.csv"



N, K, WINDOW, START = 39, 7, 384, 2


def load(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        for line, row in enumerate(csv.reader(f), 1):
            if not row or all(not v.strip() for v in row):
                continue
            try:
                numbers = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem sedam celih brojeva."
                ) from error

            if (len(numbers) != K or numbers != sorted(numbers)
                    or len(set(numbers)) != K
                    or not all(1 <= v <= N for v in numbers)):
                raise ValueError(
                    f"Red {line}: neispravna sortirana 7/39 kombinacija."
                )
            rows.append(numbers)

    return np.asarray(rows, dtype=float).reshape(-1, K)


def shape(rows):
    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), N+1),
    ))
    return np.column_stack((
        rows/(N+1),
        (np.diff(edges, axis=1)-1)/(N-K+1),
    ))


def incoming(layout, t):
    a, b = layout[t-1], layout[t-2]
    return np.r_[a, b, a*b]


def prepare(history):
    layout = shape(history)
    begin = max(START, len(history)-WINDOW)

    x = np.asarray([
        incoming(layout, t)
        for t in range(begin, len(history))
    ])
    y = history[begin:]/(N+1)

    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-8)
    a = (x-center)/scale
    mean = y.mean(axis=0)
    query = (
        incoming(layout, len(history))-center
    )/scale

    return a, y-mean, mean, query


def variational_operator(prepared, rank, regularization):
    a, y, _, _ = prepared
    gram = (
        a.T @ a
        + regularization*np.eye(a.shape[1])
    )
    factor = np.linalg.cholesky(gram)
    rhs = a.T @ y
    transformed = np.linalg.solve(factor, rhs)

    left, singular, right = np.linalg.svd(
        transformed, full_matrices=False
    )
    retained = (
        left[:, :rank]*singular[:rank]
    ) @ right[:rank]

    operator = np.linalg.solve(factor.T, retained)
    return operator, singular


def predict(prepared, rank, regularization):
    _, _, mean, query = prepared
    operator, _ = variational_operator(
        prepared, rank, regularization
    )
    target = (mean+query @ operator)*(N+1)

    if not np.all(np.isfinite(target)):
        raise ArithmeticError(
            "Neispravan varijacioni odziv."
        )
    return target


def objective(prepared, operator, regularization):
    a, y, _, _ = prepared
    residual = y-a @ operator

    return float(
        .5*np.sum(residual*residual)
        + .5*regularization*np.sum(operator*operator)
    )


def gaps(x, n=N):
    return np.diff(np.r_[0., x, float(n+1)])-1.


def cost(combo, target, n=N):
    return float(
        np.abs(np.asarray(combo)-target).sum()
        + np.abs(gaps(combo, n)-gaps(target, n)).sum()
    )


def decode(target, n=N):
    # Tacan minimum definisanog troska polozaja i razmaka.
    k = len(target)
    target_gaps = gaps(target, n)
    table = np.full((k, n+1), np.inf)
    parent = np.full((k, n+1), -1, dtype=int)

    for v in range(1, n-k+2):
        table[0, v] = (
            abs(v-target[0])
            + abs(v-1-target_gaps[0])
        )

    for j in range(1, k):
        for v in range(j+1, n-k+j+2):
            previous = np.arange(j, v)
            choices = (
                table[j-1, previous]
                + np.abs(v-previous-1-target_gaps[j])
            )
            best = int(np.argmin(choices))
            parent[j, v] = previous[best]
            table[j, v] = choices[best]+abs(v-target[j])

    ends = np.arange(k, n+1)
    scores = (
        table[k-1, ends]
        + np.abs(n-ends-target_gaps[k])
    )
    best = int(np.argmin(scores))
    v = int(ends[best])
    combo = [v]

    for j in range(k-1, 0, -1):
        v = int(parent[j, v])
        combo.append(v)

    return tuple(reversed(combo)), float(scores[best])


def hits(prediction, actual):
    return len(
        set(map(int, prediction)) & set(map(int, actual))
    )


def summarize(label, scores):
    print(
        f"{label}: prosek={np.mean(scores):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(scores, minlength=8).tolist()}",
        flush=True,
    )


def self_test():
    rows = np.asarray(list(itertools.islice(
        itertools.combinations(range(1, N+1), K),
        200,
    )), dtype=float)

    prepared = prepare(rows[:170])
    a, y, _, _ = prepared

    for rank in (2, 4, 7):
        operator, singular = variational_operator(
            prepared, rank, 10.
        )
        assert (
            np.linalg.matrix_rank(operator, tol=1e-10)
            <= rank
        )

        full, _ = variational_operator(
            prepared, K, 10.
        )
        gap = (
            objective(prepared, operator, 10.)
            - objective(prepared, full, 10.)
        )
        assert np.isclose(
            gap,
            .5*np.sum(singular[rank:]**2),
            rtol=1e-8,
            atol=1e-8,
        )

        gradient = (
            a.T @ (a @ operator-y)+10.*operator
        )
        assert np.allclose(
            operator.T @ gradient, 0, atol=1e-8
        )
        assert np.allclose(
            gradient @ operator.T, 0, atol=1e-8
        )

    target = predict(prepared, 4, 10.)
    changed = rows.copy()
    changed[170:] = [3, 8, 13, 19, 25, 31, 38]

    assert np.array_equal(
        target,
        predict(prepare(changed[:170]), 4, 10.),
    )
    assert decode(target) == decode(target)

    for target in (
        [1.8, 4.6, 7.1],
        [8., 2., 5.],
        [-2., 12., 3.],
    ):
        target = np.asarray(target)
        combo, score = decode(target, n=9)

        exact = min(
            cost(c, target, n=9)
            for c in itertools.combinations(range(1, 10), 3)
        )
        assert np.isclose(
            score, exact, rtol=1e-12, atol=1e-12
        )
        assert np.isclose(
            score, cost(combo, target, n=9)
        )

    print(
        "Self-test OK: varijaciona stacionarnost, "
        "rang, hronologija i projekcija."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default=DEFAULT)
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    rows = load(args.csv)
    if args.validation < 16 or args.test < 16:
        parser.error(
            "Validation i test moraju imati bar 16 kola."
        )

    split = len(rows)-args.test
    begin = split-args.validation
    if begin < START+128:
        parser.error(
            "Premalo istorije za zadatu validaciju i test."
        )

    parameters = tuple(itertools.product(
        (2, 4, 7), (1., 10., 100.)
    ))
    validation = [[] for _ in parameters]

    print(
        f"Ulaz: {Path(args.csv).resolve()}; "
        f"redova={len(rows)}",
        flush=True,
    )

    for t in range(begin, split):
        prepared = prepare(rows[:t])

        for j, (rank, regularization) in enumerate(parameters):
            combo, _ = decode(predict(
                prepared, rank, regularization
            ))
            validation[j].append(hits(combo, rows[t]))

    means = [
        np.mean(values) for values in validation
    ]
    best = int(np.argmax(means))
    rank, regularization = parameters[best]

    print(
        f"Validacija: redovi {begin+1}..{split}; "
        f"rang={rank}; regularizacija={regularization:g}; "
        f"prosek={means[best]:.4f}",
        flush=True,
    )

    model_scores = []
    previous_scores = []
    background_scores = []

    for t in range(split, len(rows)):
        combo, _ = decode(predict(
            prepare(rows[:t]), rank, regularization
        ))
        model_scores.append(hits(combo, rows[t]))
        previous_scores.append(hits(rows[t-1], rows[t]))

        reference, _ = decode(
            rows[max(0, t-WINDOW):t].mean(axis=0)
        )
        background_scores.append(hits(reference, rows[t]))

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize(
        "SCHWINGER V2 NUCLEAR PHYSICS",
        model_scores,
    )
    summarize("Prethodna kombinacija", previous_scores)
    summarize(
        "Pozadina bez operatora prelaza",
        background_scores,
    )

    prepared = prepare(rows)
    target = predict(prepared, rank, regularization)
    combo, score = decode(target)
    operator, singular = variational_operator(
        prepared, rank, regularization
    )

    print(
        f"Poslednji red: "
        f"{','.join(map(str, rows[-1].astype(int)))}"
    )
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"Trosak projekcije={score:.10f}; "
        "globalni minimum projekcije potvrden."
    )
    print(
        "Singularne vrednosti kanala: "
        + ",".join(
            f"{v:.6f}" for v in singular[:rank]
        )
    )
    print(
        "Operator opisuje strukturne prelaze; "
        "trosak nije verovatnoca."
    )


if __name__ == "__main__":
    main()



"""
Ulaz: /data/loto7_4698_k80.csv; redova=4698
Validacija: redovi 4587..4634; rang=7; regularizacija=1; prosek=1.2708
Test: redovi 4635..4698
SCHWINGER V2 NUCLEAR PHYSICS: prosek=1.0938; pogodaka 0..7=[17, 29, 13, 5, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Pozadina bez operatora prelaza: prosek=1.3750; pogodaka 0..7=[11, 26, 20, 6, 1, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 5,x,15,y,23,z,33
Trosak projekcije=4.6960137768; globalni minimum projekcije potvrden.
Singularne vrednosti kanala: 1.620317,0.704787,0.528804,0.423055,0.325844,0.223794,0.171863
Operator opisuje strukturne prelaze; trosak nije verovatnoca.



Ulaz: /data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: redovi 2860..2907; rang=2; regularizacija=100; prosek=1.5417
Test: redovi 2908..2971
SCHWINGER V2 NUCLEAR PHYSICS: prosek=1.1875; pogodaka 0..7=[16, 26, 16, 6, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Pozadina bez operatora prelaza: prosek=1.2812; pogodaka 0..7=[12, 29, 17, 5, 1, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 5,x,15,y,24,z,35
Trosak projekcije=5.0476844641; globalni minimum projekcije potvrden.
Singularne vrednosti kanala: 1.265194,0.591525
Operator opisuje strukturne prelaze; trosak nije verovatnoca.



Ulaz: /data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: redovi 1616..1663; rang=7; regularizacija=100; prosek=1.3333
Test: redovi 1664..1727
SCHWINGER V2 NUCLEAR PHYSICS: prosek=1.2500; pogodaka 0..7=[14, 27, 16, 7, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Pozadina bez operatora prelaza: prosek=1.2500; pogodaka 0..7=[12, 28, 20, 4, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 5,x,15,y,25,z,35
Trosak projekcije=3.2920931482; globalni minimum projekcije potvrden.
Singularne vrednosti kanala: 0.806135,0.637505,0.330350,0.310946,0.268139,0.204395,0.199590
Operator opisuje strukturne prelaze; trosak nije verovatnoca.
"""



"""
SCHWINGER V2 — NUCLEAR PHYSICS, zasnovan na varijacionom opisu ulaznih i izlaznih kanala. Model će učiti prelaz između celih konfiguracija, pa ga proveriti na istom fajlu.

Varijacioni deo je prošao proveru stacionarnosti i ograničenja broja kanala. Proverena je i projekcija na validnu kombinaciju. Sada pokrećem hronološku validaciju i test na poslednja 64 kola.

Model koristi varijacioni operator prelaza između ulaznih i izlaznih struktura, inspirisan radom Lippmanna i Schwingera o rasipanju.

Predlog za red 4699:
5, x, 15, y, 23, z, 33

Na poslednja 64 kola:
Metod	Prosečan broj pogodaka
Schwinger V2	1,0938
Prethodna kombinacija	1,2188
Pozadina bez operatora prelaza	1,3750

V2 na ovom testu nije pokazala prednost. Provereni su varijaciona stacionarnost, rang operatora, hronologija, ponovljivost i globalni minimum troška projekcije.
"""
 


"""
Julian Seymour Schwinger  
1-quantum electrodynamics.   2-nuclear physics.     3-particle physics.     
4-atomic physics       5-quantum field theory.        6-Schwinger effect
"""
