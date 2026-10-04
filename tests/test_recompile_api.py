"""HTTP 层余窗重编：成功载荷、422 前缀定位、409 余窗无解。"""

from fastapi.testclient import TestClient

from app.main import app
from app.planner import compile_plan
from app.models import CompileRequest

from .conftest import eight_slots, make_request

client = TestClient(app)

RECOMPILE_URL = "/api/momentum-plans/recompile"


def _recompile(body):
    return client.post(RECOMPILE_URL, json=body)


def _body(executed_ids, observed, slots=None, **kw):
    return {
        "original_request": make_request(slots=slots if slots is not None else eight_slots(), **kw),
        "executed_command_ids": list(executed_ids),
        "observed_momentum": list(observed),
    }


def test_recompile_success_payload():
    req = make_request(slots=eight_slots())
    plan = compile_plan(CompileRequest.model_validate(req))
    k = 3
    ids = plan["objectives"]["command_id_sequence"][:k]
    observed = plan["momentums"][k - 1]

    r = _recompile(_body(ids, observed))
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["feasible"] is True
    assert data["slot_count"] == 8
    assert data["replanned_from_slot"] == k + 1
    assert tuple(data["observed_momentum"]) == tuple(observed)
    assert [s["slot"] for s in data["selected"]] == [4, 5, 6, 7, 8]
    assert len(data["momentums"]) == 5
    # 整窗目标：编号序列 8 条。
    assert len(data["objectives"]["command_id_sequence"]) == 8
    assert set(data) >= {
        "selected",
        "momentums",
        "final_momentum",
        "objectives",
        "slot_count",
        "replanned_from_slot",
        "observed_momentum",
    }
    sel = data["selected"][0]
    assert set(sel) == {"slot", "command_id", "mode", "correction", "energy"}


def test_recompile_mid_prefix_keeps_executed_ids():
    req = make_request(slots=eight_slots())
    plan = compile_plan(CompileRequest.model_validate(req))
    k = 5
    ids = plan["objectives"]["command_id_sequence"][:k]
    observed = plan["momentums"][k - 1]

    r = _recompile(_body(ids, observed))
    assert r.status_code == 200, r.text
    data = r.json()
    # 编号序列以前缀开头，后缀只重选剩余时隙。
    full = data["objectives"]["command_id_sequence"]
    assert full[:k] == ids
    assert len(full) == 8
    assert data["replanned_from_slot"] == 6
    assert [s["slot"] for s in data["selected"]] == [6, 7, 8]


def test_missing_executed_id_is_located():
    body = _body([1], (0, 0))
    # 时隙 1 的候选编号为 {1,2}，3 不属于该时隙。
    body["executed_command_ids"] = [3]
    r = _recompile(body)
    assert r.status_code == 422
    data = r.json()
    assert data["error"] == "invalid_request"
    assert data["field"] == "executed_command_ids[0]"


def test_id_belongs_to_wrong_slot_is_located():
    # 编号 2 属于时隙 1，但把它放到第 2 个前缀位置（时隙 2 也恰好有 id=2…），
    # 因此构造一个时隙 2 不含该编号的场景。
    slots = eight_slots()
    # 时隙 2 的指令编号改为 10/20。
    for c in slots[1]["commands"]:
        c["id"] += 9
    body = _body([2, 20], (0, 0), slots=slots)
    body["executed_command_ids"] = [1, 2]  # id=2 不属于时隙 2
    r = _recompile(body)
    assert r.status_code == 422
    assert r.json()["field"] == "executed_command_ids[1]"


def test_repeated_id_across_slots_is_allowed():
    """编号仅在时隙内唯一：跨时隙复用同一编号（如并列裁决产出 [3]*8）合法。"""
    slots = [
        {
            "disturbance": [0, 0],
            "commands": [
                {"id": 3, "mode": "A", "correction": [0, 0], "energy": 1.0},
                {"id": 7, "mode": "A", "correction": [0, 0], "energy": 1.0},
            ],
        }
    ] * 8
    body = _body([3, 3, 3], (0, 0), slots=slots)
    r = _recompile(body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["objectives"]["command_id_sequence"] == [3] * 8


def test_empty_prefix_rejected():
    body = _body([], (0, 0))
    r = _recompile(body)
    assert r.status_code == 422


def test_prefix_covering_all_slots_rejected():
    body = _body([1] * 8, (0, 0))
    r = _recompile(body)
    assert r.status_code == 422
    assert r.json()["field"] == "executed_command_ids"


def test_observed_outside_safety_is_located():
    body = _body([1], (101, 0))
    r = _recompile(body)
    assert r.status_code == 422
    assert r.json()["field"] == "observed_momentum"


def test_original_request_still_semantically_validated():
    body = _body([1], (0, 0))
    body["original_request"]["initial_momentum"] = [101, 0]
    r = _recompile(body)
    assert r.status_code == 422
    # 字段定位带上 original_request 前缀，与顶层字段区分。
    assert r.json()["field"] == "original_request.initial_momentum"


def test_unknown_field_rejected():
    body = _body([1], (0, 0))
    body["nonsense"] = 1
    r = _recompile(body)
    assert r.status_code == 422


def test_no_safe_suffix_returns_409_with_original_slot_numbering():
    # 每隙 x 固定 +1，安全域 x∈[0,3]：观测 (3,0) 后第 4 隙必越界。
    slots = [
        {
            "disturbance": [1, 0],
            "commands": [
                {"id": 1, "mode": "A", "correction": [0, 0], "energy": 1.0},
                {"id": 2, "mode": "B", "correction": [0, 0], "energy": 2.0},
            ],
        }
    ] * 8
    body = _body(
        [1, 1, 1],
        (3, 0),
        slots=slots,
        safety={"x_min": 0, "x_max": 3, "y_min": -1, "y_max": 1},
        target={"x_min": 3, "x_max": 3, "y_min": 0, "y_max": 0},
    )
    r = _recompile(body)
    assert r.status_code == 409
    data = r.json()
    assert data["error"] == "no_feasible_plan"
    assert data["last_reachable_slot"] == 3
    assert data["reachable_states"] == 1


def test_compile_endpoint_unchanged():
    r = client.post(
        "/api/momentum-plans/compile", json=make_request(slots=eight_slots())
    )
    assert r.status_code == 200
    data = r.json()
    assert "replanned_from_slot" not in data
    assert "observed_momentum" not in data
