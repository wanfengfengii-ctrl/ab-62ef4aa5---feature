"""余窗重编规划器：前缀固定、观测续算、整窗目标与无解定位。"""

import pytest

from app.models import Command, CompileRequest
from app.planner import InfeasiblePlanError, compile_plan, recompile_plan

from .conftest import brute_force_recompile, eight_slots, make_request, slot


def _prefix_cmds(req: dict, ids: list[int]) -> list[Command]:
    cmds = []
    for i, cid in enumerate(ids):
        raw = next(c for c in req["slots"][i]["commands"] if c["id"] == cid)
        cmds.append(Command.model_validate(raw))
    return cmds


def _recompile(req, ids, observed):
    return recompile_plan(
        CompileRequest.model_validate(req),
        _prefix_cmds(req, ids),
        tuple(observed),
        replanned_from_slot=len(ids) + 1,
    )


def test_recompile_shape_and_markers():
    req = make_request(slots=eight_slots())
    plan = compile_plan(CompileRequest.model_validate(req))
    ids = plan["objectives"]["command_id_sequence"][:3]

    # 观测动量取原计划第 3 隙末动量（重编应复现原后缀）。
    observed = plan["momentums"][2]
    result = _recompile(req, ids, observed)

    assert result["feasible"] is True
    assert result["slot_count"] == 8
    assert result["replanned_from_slot"] == 4
    assert tuple(result["observed_momentum"]) == tuple(observed)
    # selected/momentums 只覆盖剩余 5 个时隙，槽位编号沿用原窗口。
    assert [s["slot"] for s in result["selected"]] == [4, 5, 6, 7, 8]
    assert len(result["momentums"]) == 5


def test_recompile_with_matching_observation_reproduces_plan():
    """观测动量与原计划前缀末态一致时，后缀选择与原计划完全相同。"""
    req = make_request(slots=eight_slots())
    plan = compile_plan(CompileRequest.model_validate(req))
    for k in (1, 3, 7):
        ids = plan["objectives"]["command_id_sequence"][:k]
        observed = plan["momentums"][k - 1]
        result = _recompile(req, ids, observed)

        suffix_ids = plan["objectives"]["command_id_sequence"][k:]
        assert result["objectives"]["command_id_sequence"] == (
            plan["objectives"]["command_id_sequence"]
        )
        assert [s["command_id"] for s in result["selected"]] == suffix_ids
        assert result["objectives"]["total_energy"] == pytest.approx(
            plan["objectives"]["total_energy"]
        )
        assert result["objectives"]["mode_switches"] == plan["objectives"]["mode_switches"]


def test_objectives_cover_full_window():
    """能耗、切换次数、编号序列覆盖完整窗口，交界切换计入。"""
    # 同能耗下切换次数才是裁决量：A/B 能耗相同。
    slots = [
        slot((0, 0), (1, "A", (0, 0), 1.0), (2, "B", (0, 0), 1.0)),
        slot((0, 0), (1, "A", (0, 0), 1.0), (2, "B", (0, 0), 1.0)),
    ] + [
        slot((0, 0), (1, "A", (0, 0), 1.0), (2, "B", (0, 0), 1.0))
        for _ in range(6)
    ]
    req = make_request(slots=slots)
    # 前缀 A 然后 B：前缀内部一次切换；后缀全 B 时交界处 B->B 无切换，
    # 而后缀全 A 会多出一次交界切换，故三级裁决选全 B，总切换 1 次。
    result = _recompile(req, [1, 2], (0, 0))
    assert result["objectives"]["mode_switches"] == 1
    assert result["objectives"]["command_id_sequence"] == [1, 2] + [2] * 6
    assert result["objectives"]["total_energy"] == pytest.approx(8.0)


def test_boundary_mode_switch_is_counted():
    """前后缀交界模式不同也要计一次切换。"""
    slots = [
        slot((0, 0), (1, "A", (0, 0), 1.0), (2, "B", (0, 0), 1.0)),
    ] * 8
    req = make_request(slots=slots)
    # 前缀全 A（3 隙，零内部切换），后缀在能耗并列下应选 A 以避免交界切换。
    result = _recompile(req, [1, 1, 1], (0, 0))
    assert result["objectives"]["mode_switches"] == 0

    # 迫使后缀首条模式为 B：该场景下 A/B 同能耗同修正，字典序 id=1（A）更小，
    # 因此构造后缀首隙只有 B 能保持安全的场景来直接核对交界计数。
    slots2 = [
        slot((0, 0), (1, "A", (0, 0), 1.0), (2, "B", (0, 0), 1.0))
        for _ in range(3)
    ] + [
        slot((0, 0), (3, "B", (0, 0), 1.0), (4, "C", (1, 0), 1.0)),
    ] + [
        slot((0, 0), (1, "A", (0, 0), 1.0), (2, "B", (0, 0), 1.0))
        for _ in range(4)
    ]
    req2 = make_request(slots=slots2)
    result2 = _recompile(req2, [1, 1, 1], (0, 0))
    # 第 4 隙只能在 B/C 中选，A->B 交界切换 1 次。
    assert result2["objectives"]["command_id_sequence"][3] in (3, 4)
    assert result2["objectives"]["mode_switches"] >= 1


def test_suffix_evolution_starts_from_observed():
    """后缀动量从观测动量继续，且末态进入原目标域。"""
    req = make_request(
        target={"x_min": -2, "x_max": 2, "y_min": -2, "y_max": 2},
        slots=eight_slots(),
    )
    observed = (1, 1)
    result = _recompile(req, [1, 1], observed)

    state = observed
    target = req["target_region"]
    suffix_slots = req["slots"][2:]
    for sel, m, s in zip(result["selected"], result["momentums"], suffix_slots):
        d = s["disturbance"]
        c = sel["correction"]
        state = (state[0] + d[0] + c[0], state[1] + d[1] + c[1])
        assert tuple(m) == state
    fx, fy = result["final_momentum"]
    assert target["x_min"] <= fx <= target["x_max"]
    assert target["y_min"] <= fy <= target["y_max"]


def test_deviated_observation_changes_suffix():
    """实测偏离原轨迹时，后缀会重新选择以从观测状态进入目标域。"""
    req = make_request(slots=eight_slots())
    plan = compile_plan(CompileRequest.model_validate(req))
    ids = plan["objectives"]["command_id_sequence"][:2]
    # 偏离原计划第 2 隙末态（换成另一个安全域内的点）。
    planned_obs = tuple(plan["momentums"][1])
    deviated = (planned_obs[0] + 2, planned_obs[1])
    assert deviated != planned_obs

    result = _recompile(req, ids, deviated)
    assert tuple(result["objectives"]["command_id_sequence"][:2]) == tuple(ids)
    # 从偏离状态重编仍须可解（宽安全域 + 宽目标域）。
    assert result["replanned_from_slot"] == 3


def test_no_safe_suffix_reports_original_window_slot():
    """合法前缀但观测状态无安全后缀：409，编号沿用原窗口。"""
    # 每隙 x 固定 +1；观测动量已在安全域右边界，下一隙必越界。
    slots = [
        slot((1, 0), (1, "A", (0, 0), 1.0), (2, "B", (0, 0), 2.0))
    ] * 8
    req = make_request(
        safety={"x_min": 0, "x_max": 3, "y_min": -1, "y_max": 1},
        target={"x_min": 3, "x_max": 3, "y_min": 0, "y_max": 0},
        slots=slots,
    )
    # 已执行 3 隙（x=3 恰在边界），第 4 隙必然 +1 越界。
    with pytest.raises(InfeasiblePlanError) as exc:
        _recompile(req, [1, 1, 1], (3, 0))
    assert exc.value.reason == "no_safe_path"
    assert exc.value.last_reachable_slot == 3  # 原窗口编号
    assert exc.value.reachable_states == 1


def test_target_unreachable_from_observed():
    """后缀全程安全但无法从观测状态进入目标域。"""
    slots = [
        slot((1, 0), (1, "A", (0, 0), 1.0), (2, "B", (0, 0), 1.0))
    ] * 8
    req = make_request(
        safety={"x_min": 0, "x_max": 50, "y_min": 0, "y_max": 50},
        target={"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0},
        slots=slots,
    )
    # 前缀执行 2 隙后观测到 (5,0)（偏离标称 (2,0)），末态必为 (11,0)。
    with pytest.raises(InfeasiblePlanError) as exc:
        _recompile(req, [1, 1], (5, 0))
    assert exc.value.reason == "target_unreachable"
    assert exc.value.last_reachable_slot == 8
    assert exc.value.reachable_states == 1


@pytest.mark.parametrize("seed", range(40))
def test_recompile_matches_brute_force(seed):
    """随机对拍：固定前缀 + 观测动量，DP 与独立穷举一致。"""
    import random

    from tests.test_planner_property import _random_request

    rng = random.Random(1000 + seed)
    req = _random_request(rng, 8, rng.randint(2, 3))
    s, t = req["safety_region"], req["target_region"]
    t["x_min"] = max(t["x_min"], s["x_min"])
    t["x_max"] = min(t["x_max"], s["x_max"])
    t["y_min"] = max(t["y_min"], s["y_min"])
    t["y_max"] = min(t["y_max"], s["y_max"])

    k = rng.randint(1, 7)
    ids = [rng.choice([c["id"] for c in req["slots"][i]["commands"]]) for i in range(k)]
    observed = [rng.randint(s["x_min"], s["x_max"]), rng.randint(s["y_min"], s["y_max"])]

    expected = brute_force_recompile(req, ids, observed)
    try:
        result = _recompile(req, ids, observed)
    except InfeasiblePlanError:
        assert expected is None
        return

    assert expected is not None
    got = (
        result["objectives"]["total_energy"],
        result["objectives"]["mode_switches"],
        tuple(result["objectives"]["command_id_sequence"]),
    )
    assert got == expected
