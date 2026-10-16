import numpy as np
import pytest

from src.microgrid import HybridMicroGrid, MicroGrid, generate_demand_csv, load_demand_csv, prompt_positive_float


def test_solve_day_matches_hand_solution():
    # x=10, y=5 → D1 = 40, D2 = 45
    np.testing.assert_allclose(MicroGrid().solve_day(40, 45), [10, 5])


def test_determinant_and_condition():
    g = MicroGrid()
    assert g.determinant() == pytest.approx(-5)
    assert 1 < g.condition_number() < 10


def test_loop_equals_vectorised(tmp_path):
    generate_demand_csv(tmp_path / "d.csv", days=30)
    D = load_demand_csv(tmp_path / "d.csv")
    g = MicroGrid()
    np.testing.assert_allclose(g.solve_loop(D), g.solve_vectorised(D))


def test_infeasible_day_handled_by_nnls():
    g = MicroGrid()
    D = np.array([[40, 10], [45, 40]])   # day 2: x = 14, y = -16 → infeasible
    sol, resid = g.solve_nonnegative(D)
    assert g.infeasible_days(g.solve_vectorised(D)).tolist() == [False, True]
    assert np.all(sol >= 0) and resid[0] == 0 and resid[1] > 0


def test_negative_demand_rejected():
    with pytest.raises(ValueError):
        MicroGrid().solve_day(-1, 5)


def test_singular_system_rejected():
    dependent = HybridMicroGrid([[3, 2, 1], [4, 1, 2], [7, 3, 3]])   # row3 = row1 + row2
    with pytest.raises(np.linalg.LinAlgError):
        dependent.solve_day(10, 10, 20)


def test_prompt_reprompts_until_valid(capsys):
    answers = iter(["", "abc", "-5", "12.5"])
    assert prompt_positive_float("D1: ", input_func=lambda _: next(answers)) == 12.5
    assert capsys.readouterr().out.count("✗") == 3
