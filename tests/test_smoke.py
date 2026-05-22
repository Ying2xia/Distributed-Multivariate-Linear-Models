from __future__ import annotations

import numpy as np

from dmlm.data import coefficient_matrix, generate_distributed_simulation
from dmlm.diagnostics import contaminate_response, detection_summary, observation_influence, worker_influence
from dmlm.estimators import fit_dmlm_s, fit_first_worker, fit_local_average, fit_pilot_only, fit_workers_full
from dmlm.metrics import relative_error


def test_estimators_return_finite_shapes() -> None:
    rng = np.random.default_rng(123)
    p, m = 5, 3
    B0 = coefficient_matrix(p, m, "S1", rng)
    workers, _, _ = generate_distributed_simulation(
        N=500,
        K=5,
        p=p,
        m=m,
        structure="S1",
        strategy="C",
        rho_x=0.5,
        rho_y=0.6,
        rng=rng,
        B0=B0,
    )
    seed = 456
    results = [
        fit_workers_full(workers),
        fit_local_average(workers),
        fit_pilot_only(workers, rng=np.random.default_rng(seed), pilot_proportion=0.12),
        fit_first_worker(workers),
        fit_dmlm_s(workers, rng=np.random.default_rng(seed), pilot_proportion=0.12, rounds=3),
    ]
    for result in results:
        assert result.B.shape == (p, m)
        assert result.Sigma.shape == (m, m)
        assert np.all(np.isfinite(result.B))
        assert np.all(np.isfinite(result.Sigma))


def test_dmlm_moves_toward_gmlm_from_pilot() -> None:
    rng = np.random.default_rng(321)
    p, m = 5, 3
    B0 = coefficient_matrix(p, m, "S1", rng)
    workers, _, _ = generate_distributed_simulation(
        N=800,
        K=8,
        p=p,
        m=m,
        structure="S1",
        strategy="H",
        rho_x=0.5,
        rho_y=0.6,
        rng=rng,
        B0=B0,
    )
    seed = 654
    gmlm = fit_workers_full(workers)
    pilot = fit_pilot_only(workers, rng=np.random.default_rng(seed), pilot_proportion=0.15)
    dmlm = fit_dmlm_s(workers, rng=np.random.default_rng(seed), pilot_proportion=0.15, rounds=5)
    assert relative_error(dmlm.B, gmlm.B) <= relative_error(pilot.B, gmlm.B)


def test_influence_outputs_rankings() -> None:
    rng = np.random.default_rng(999)
    p, m = 5, 3
    B0 = coefficient_matrix(p, m, "S1", rng)
    workers, _, _ = generate_distributed_simulation(
        N=300,
        K=5,
        p=p,
        m=m,
        structure="S1",
        strategy="C",
        rho_x=0.5,
        rho_y=0.6,
        rng=rng,
        B0=B0,
    )
    contaminated = {10, 20, 30}
    contaminated_workers = contaminate_response(workers, contaminated, shift=8.0)
    result = fit_dmlm_s(contaminated_workers, rng=np.random.default_rng(1000), pilot_proportion=0.15, rounds=3)
    obs = observation_influence(contaminated_workers, result.B, hessian_x=result.extra["pilot_hessian"])
    wk = worker_influence(contaminated_workers, result.B, hessian_x=result.extra["pilot_hessian"])
    summary = detection_summary(obs, wk, contaminated)
    assert {"Hit@5", "Hit@10", "AUC"}.issubset(summary)
    assert len(obs) == 300
    assert len(wk) == 5

