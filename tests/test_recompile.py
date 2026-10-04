"""余窗重编规划器：固定前缀、观测点续演、三级目标与无解定位。"""

import random

import pytest

from app.models import CompileRequest
from app.planner import InfeasiblePlanError, recompile_plan

from .conftest import brute_force_recompile_best, eight_slots, make_request, slot


def _recompile(req, executed, observed):
    return recompile_plan(
        CompileRequest.model_validate(req),
        executed,
        tuple(observed),
    )


def test_recompile_keeps_executed_prefix():
    """前 k 隙指令编号原样保留，重编只作用于剩余时隙。"""
    req = make_request(slots=eight_slots())
    result = _recompile(req, [1, 2, 1], observed=(0, 0))
    ids = result["objectives"]["command_id_sequence"]
    assert len(ids) == 8
    assert ids[:3] == [1, 2, 1]
    assert result["replanned_from_slot"] == 4
    assert tuple(result["observed_momentum"]) == (0, 0)
    assert result["slot_count"] == 8
    # selected 保留完整窗口且逐隙编号正确。
    assert [s["slot"] for s in result["selected"]] == list(range(1, 9))
    assert [s["command_id"] for s in result["selected"][:3]] == [1, 2, 1]


def test_recompile_suffix_evolution_starts_at_observation():
    """后缀动量从观测动量继续：p_{k+1} = 观测 + 扰动 + 修正。"""
    req = make_request(slots=eight_slots())
    observed = (5, -3)
    result = _recompile(req, [2, 1], observed=observed)

    assert tuple(result["momentums"][1]) == observed  # 第 k 隙末态为观测动量
    state = observed
    safety = req["safety_region"]
    for i in range(2, 8):  # 第 3 隙起为后缀
        sel = result["selected"][i]
        d = req["slots"][i]["disturbance"]
        state = (
            state[0] + d[0] + sel["correction"][0],
            state[1] + d[1] + sel["correction"][1],
        )
        assert tuple(result["momentums"][i]) == state
        assert safety["x_min"] <= state[0] <= safety["x_max"]
        assert safety["y_min"] <= state[1] <= safety["y_max"]
    # 末态仍进入原目标域。
    assert tuple(result["final_momentum"]) == state


def test_recompile_final_state_in_original_target():
    req = make_request(
        target={"x_min": -2, "x_max": 2, "y_min": -2, "y_max": 2},
        slots=eight_slots(),
    )
    result = _recompile(req, [1, 1, 1], observed=(1, 1))
    fx, fy = result["final_momentum"]
    assert -2 <= fx <= 2 and -2 <= fy <= 2


def test_recompile_prefix_costs_included_with_junction_switch():
    """总能耗与切换数覆盖完整窗口，前后缀交界模式不同也计一次。"""
    # 后缀 A（id=1）更省能耗，因此即使与前缀末模式 B 不同也必须切换：
    # 能耗优先级高于切换次数。
    slots = [
        slot((0, 0), (1, "A", (0, 0), 0.5), (2, "B", (0, 0), 1.0)),
    ] * 8
    req = make_request(slots=slots)
    result = _recompile(req, [2, 2, 2], observed=(0, 0))
    ids = result["objectives"]["command_id_sequence"]
    assert ids == [2, 2, 2, 1, 1, 1, 1, 1]
    # 交界 B->A 恰好一次切换；前缀能耗 3.0 + 后缀能耗 2.5。
    assert result["objectives"]["mode_switches"] == 1
    assert result["objectives"]["total_energy"] == pytest.approx(5.5)


def test_recompile_no_junction_switch_when_modes_match():
    slots = [
        slot((0, 0), (1, "A", (0, 0), 1.0), (2, "B", (0, 0), 1.0)),
    ] * 8
    req = make_request(slots=slots)
    result = _recompile(req, [1, 1], observed=(0, 0))
    assert result["objectives"]["mode_switches"] == 0
    assert result["objectives"]["command_id_sequence"] == [1] * 8


def test_recompile_deviation_changes_suffix_choice():
    """观测偏差改变后缀最优：同一前缀，观测点不同则后缀选择不同。

    每隙扰动 0：id=1 模式 A 修正 (-1,0) 能耗 1；id=2 模式 B 修正 (0,0) 能耗 5。
    安全域 x∈[0,3]，目标 x=0。固定前缀 [2,2,2]（重放停留原点）：
    - 观测仍在原点时，任何 id=1 都会越界，后缀只能全选 id=2；
    - 实测漂移到 x=3 时，后缀须恰好三次 id=1、两次 id=2 回到原点；
      能耗固定 13，二级目标最少切换决定排列为 [2,2,1,1,1]（仅交界一次切换）。
    """
    slots = [
        slot((0, 0), (1, "A", (-1, 0), 1.0), (2, "B", (0, 0), 5.0)),
    ] * 8
    req = make_request(
        safety={"x_min": 0, "x_max": 3, "y_min": -1, "y_max": 1},
        target={"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0},
        slots=slots,
    )

    at_origin = _recompile(req, [2, 2, 2], observed=(0, 0))
    assert at_origin["objectives"]["command_id_sequence"] == [2] * 8
    assert at_origin["objectives"]["mode_switches"] == 0
    assert at_origin["objectives"]["total_energy"] == pytest.approx(40.0)

    drifted = _recompile(req, [2, 2, 2], observed=(3, 0))
    ids = drifted["objectives"]["command_id_sequence"]
    assert ids == [2, 2, 2, 2, 2, 1, 1, 1]
    assert drifted["objectives"]["total_energy"] == pytest.approx(28.0)  # 15 + 13
    # 后缀中段 B->A 恰好一次切换（前缀 B 与后缀首个 B 衔接不切换）。
    assert drifted["objectives"]["mode_switches"] == 1
    # 后缀逐隙动量安全且末态回原点。
    assert [tuple(m) for m in drifted["momentums"][3:]] == [
        (3, 0), (3, 0), (2, 0), (1, 0), (0, 0),
    ]
    assert tuple(drifted["final_momentum"]) == (0, 0)


def test_recompile_no_safe_suffix_returns_409_diagnostics():
    """合法前缀但观测状态下余窗第一步即被切断：最后可达时隙为 k。"""
    # 后缀时隙扰动 +1 且所有修正非负，观测已在上边界。
    slots = [
        slot((0, 0), (1, "A", (0, 0), 1.0), (2, "B", (0, 0), 2.0)),
    ] * 4 + [
        slot((1, 0), (1, "A", (0, 0), 1.0), (2, "B", (1, 0), 1.0)),
    ] * 4
    req = make_request(
        safety={"x_min": 0, "x_max": 2, "y_min": -1, "y_max": 1},
        target={"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 1},
        slots=slots,
    )
    # 前四隙 0 扰动，观测 x=2 合法；第 5 隙任何指令结果 x>=3 越界。
    with pytest.raises(InfeasiblePlanError) as exc:
        _recompile(req, [1, 1, 1, 1], observed=(2, 0))
    assert exc.value.reason == "no_safe_path"
    assert exc.value.last_reachable_slot == 4  # 原窗口编号
    assert exc.value.reachable_states == 1


def test_recompile_target_unreachable_reports_full_window():
    """后缀安全域内可达但无人进入目标：最后可达时隙为总时隙数。"""
    # 后缀全部 +1 无负修正，观测 x=0 后只能单调向右，无法回到原点目标。
    slots = [
        slot((0, 0), (1, "A", (0, 0), 1.0), (2, "B", (0, 0), 1.0)),
    ] * 4 + [
        slot((1, 0), (1, "A", (0, 0), 1.0), (2, "B", (0, 0), 1.0)),
    ] * 4
    req = make_request(
        safety={"x_min": 0, "x_max": 50, "y_min": -1, "y_max": 1},
        target={"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0},
        slots=slots,
    )
    with pytest.raises(InfeasiblePlanError) as exc:
        _recompile(req, [1, 1, 1, 1], observed=(0, 0))
    assert exc.value.reason == "target_unreachable"
    assert exc.value.last_reachable_slot == 8
    assert exc.value.reachable_states == 1  # 唯一动量 (4,0)


def test_recompile_k_at_maximum_bound():
    """k = T-1（仅剩一个余窗时隙）也合法。"""
    req = make_request(slots=eight_slots())
    result = _recompile(req, [1] * 7, observed=(0, 0))
    assert result["replanned_from_slot"] == 8
    assert len(result["objectives"]["command_id_sequence"]) == 8


def _random_request(rng: random.Random, n_slots: int, n_cmds: int) -> dict:
    modes = ["A", "B", "C"]
    slots = []
    for _ in range(n_slots):
        ids = rng.sample(range(1, 100), n_cmds)
        commands = []
        for cid in ids:
            commands.append(
                {
                    "id": cid,
                    "mode": rng.choice(modes),
                    "correction": [rng.randint(-3, 3), rng.randint(-3, 3)],
                    "energy": rng.choice([0.0, 0.5, 1.0, 2.0, 5.0]),
                }
            )
        slots.append(
            {
                "disturbance": [rng.randint(-2, 2), rng.randint(-2, 2)],
                "commands": commands,
            }
        )
    bound = rng.randint(4, 12)
    return make_request(
        initial=(rng.randint(-2, 2), rng.randint(-2, 2)),
        safety={
            "x_min": -bound,
            "x_max": bound,
            "y_min": -bound,
            "y_max": bound,
        },
        target={
            "x_min": rng.randint(-bound, 0),
            "x_max": rng.randint(0, bound),
            "y_min": rng.randint(-bound, 0),
            "y_max": rng.randint(0, bound),
        },
        slots=slots,
    )


@pytest.mark.parametrize("seed", range(80))
def test_recompile_matches_brute_force(seed):
    """随机对拍：固定前缀 + 随机观测动量，DP 与独立穷举参考完全一致。"""
    rng = random.Random(seed)
    req = _random_request(rng, 8, rng.randint(2, 3))

    # 与整窗对拍相同的输入归一化：目标域包含于安全域、初始动量在安全域内。
    s, t = req["safety_region"], req["target_region"]
    t["x_min"] = max(t["x_min"], s["x_min"])
    t["x_max"] = min(t["x_max"], s["x_max"])
    t["y_min"] = max(t["y_min"], s["y_min"])
    t["y_max"] = min(t["y_max"], s["y_max"])
    req["initial_momentum"] = [
        min(max(req["initial_momentum"][0], s["x_min"]), s["x_max"]),
        min(max(req["initial_momentum"][1], s["y_min"]), s["y_max"]),
    ]

    k = rng.randint(1, 7)
    # 前缀编号依次取各时隙内随机一条合法指令。
    executed = [
        rng.choice([c["id"] for c in req["slots"][i]["commands"]])
        for i in range(k)
    ]
    observed = (rng.randint(s["x_min"], s["x_max"]), rng.randint(s["y_min"], s["y_max"]))

    expected = brute_force_recompile_best(req, executed, observed)
    try:
        result = _recompile(req, executed, observed)
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
    # 前缀与观测点衔接也在对拍中固定核对。
    assert got[2][:k] == tuple(executed)
    assert tuple(result["observed_momentum"]) == observed
    assert result["replanned_from_slot"] == k + 1
