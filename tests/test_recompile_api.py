"""余窗重编 HTTP 层：成功结构、422 前缀/观测校验定位、409 无安全后缀。"""

from fastapi.testclient import TestClient

from app.main import app

from .conftest import eight_slots, make_request

client = TestClient(app)
RECOMPILE_URL = "/api/momentum-plans/recompile"
COMPILE_URL = "/api/momentum-plans/compile"


def _body(slots=None, executed=None, observed=(0, 0), **kw):
    req = make_request(slots=slots if slots is not None else eight_slots(), **kw)
    return {
        "request": req,
        "executed_command_ids": executed if executed is not None else [1, 2, 1],
        "observed_momentum": list(observed),
    }


def _recompile(body):
    return client.post(RECOMPILE_URL, json=body)


def test_recompile_success_payload():
    r = _recompile(_body(executed=[1, 2, 1], observed=(0, 0)))
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["feasible"] is True
    assert data["slot_count"] == 8
    assert data["replanned_from_slot"] == 4
    assert data["observed_momentum"] == [0, 0]
    # 沿用原计划结构。
    assert set(data) >= {
        "selected",
        "momentums",
        "final_momentum",
        "objectives",
    }
    assert len(data["selected"]) == 8
    assert len(data["momentums"]) == 8
    # 前缀固定。
    assert [s["command_id"] for s in data["selected"][:3]] == [1, 2, 1]
    assert [s["slot"] for s in data["selected"]] == list(range(1, 9))
    ids = data["objectives"]["command_id_sequence"]
    assert ids[:3] == [1, 2, 1] and len(ids) == 8


def test_recompile_suffix_evolves_from_observation():
    """后缀第一隙动量 = 观测动量 + 扰动 + 修正。"""
    r = _recompile(_body(executed=[2, 2], observed=(4, -2)))
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["momentums"][1] == [4, -2]  # 第 k 隙末态即观测值
    sel = data["selected"][2]
    d = eight_slots()[2]["disturbance"]
    expect = [
        4 + d[0] + sel["correction"][0],
        -2 + d[1] + sel["correction"][1],
    ]
    assert data["momentums"][2] == expect
    assert data["final_momentum"] == data["momentums"][-1]


def test_recompile_keeps_original_compile_endpoint():
    """原编译接口契约不变。"""
    body = make_request(slots=eight_slots())
    r = client.post(COMPILE_URL, json=body)
    assert r.status_code == 200
    assert set(r.json()) == {
        "feasible",
        "slot_count",
        "selected",
        "momentums",
        "final_momentum",
        "objectives",
    }


def test_empty_executed_prefix_is_422():
    body = _body(executed=[])
    r = _recompile(body)
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_request"


def test_executed_prefix_covering_all_slots_is_422():
    body = _body(executed=[1] * 8)
    r = _recompile(body)
    assert r.status_code == 422
    assert r.json()["field"] == "executed_command_ids"


def test_missing_prefix_id_is_422():
    # 编号 99 不属于第 1 时隙（候选编号 1/2）。
    body = _body(executed=[99, 1])
    r = _recompile(body)
    assert r.status_code == 422
    data = r.json()
    assert data["error"] == "invalid_request"
    assert data["field"] == "executed_command_ids[0]"


def test_prefix_id_from_wrong_slot_is_422():
    """编号存在于窗口但不属于被上报到的那个时隙 → 定位该位置 422。"""
    slots = eight_slots()
    # 第 1 时隙编号 {1,2}，第 2 时隙编号 {3,4}。
    slots[1]["commands"][0]["id"] = 3
    slots[1]["commands"][1]["id"] = 4
    body = _body(slots=slots, executed=[1, 9])  # 9 不属于第 2 时隙
    r = _recompile(body)
    assert r.status_code == 422
    assert r.json()["field"] == "executed_command_ids[1]"


def test_prefix_must_cover_first_slots_in_order():
    """编号合法但若跳过前隙（首位置使用后隙编号）按字段 422。"""
    slots = eight_slots()
    # 令第 1 时隙编号 {5,6}、第 2 时隙编号 {1,2}，此时 executed 首项 1
    # 并不属于第 1 时隙。
    slots[0]["commands"][0]["id"] = 5
    slots[0]["commands"][1]["id"] = 6
    body = _body(slots=slots, executed=[1, 2])
    r = _recompile(body)
    assert r.status_code == 422
    assert r.json()["field"] == "executed_command_ids[0]"


def test_same_id_reused_across_slots_is_legal_prefix():
    """编号仅时隙内唯一：跨时隙重复选择同一编号（如 [1,1,1]）是正常前缀。"""
    r = _recompile(_body(executed=[1, 1, 1, 1, 1, 1, 1]))
    assert r.status_code == 200, r.text
    assert r.json()["objectives"]["command_id_sequence"][:7] == [1] * 7


def test_duplicate_candidate_id_in_slot_is_422():
    """同一时隙内候选指令编号重复（编号重复定义）仍按字段 422。"""
    slots = eight_slots()
    slots[2]["commands"][1]["id"] = slots[2]["commands"][0]["id"]
    body = _body(slots=slots, executed=[1, 2, 1])
    r = _recompile(body)
    assert r.status_code == 422
    assert r.json()["field"] == "request.slots[2].commands[1].id"


def test_observed_momentum_outside_safety_is_422():
    body = _body(executed=[1], observed=(101, 0))
    r = _recompile(body)
    assert r.status_code == 422
    assert r.json()["field"] == "observed_momentum"


def test_nested_request_semantic_error_is_prefixed():
    """嵌套原请求的语义矛盾仍 422，字段路径带 request. 前缀。"""
    body = _body(executed=[1], observed=(0, 0))
    body["request"]["initial_momentum"] = [999, 0]
    r = _recompile(body)
    assert r.status_code == 422
    assert r.json()["field"] == "request.initial_momentum"


def test_nested_request_type_error_is_located():
    body = _body()
    body["request"]["slots"] = body["request"]["slots"][:3]  # 少于 8 隙
    r = _recompile(body)
    assert r.status_code == 422
    assert any("request.slots" in d["field"] for d in r.json()["details"])


def test_unknown_field_rejected():
    body = _body()
    body["nonsense"] = 1
    r = _recompile(body)
    assert r.status_code == 422


def test_no_safe_suffix_is_409_with_original_window_slot():
    """合法前缀 + 合法观测，但余窗第一步即被安全域切断 → 409。

    last_reachable_slot 以原窗口编号报告（等于前缀长度 k）。
    """
    slots = [
        {
            "disturbance": [0, 0],
            "commands": [
                {"id": 1, "mode": "A", "correction": [0, 0], "energy": 1.0},
                {"id": 2, "mode": "B", "correction": [0, 0], "energy": 2.0},
            ],
        }
    ] * 4 + [
        {
            "disturbance": [1, 0],
            "commands": [
                {"id": 1, "mode": "A", "correction": [0, 0], "energy": 1.0},
                {"id": 2, "mode": "B", "correction": [1, 0], "energy": 1.0},
            ],
        }
    ] * 4
    body = _body(
        slots=slots,
        executed=[1, 1, 1, 1],
        observed=(2, 0),
        **{
            "safety": {"x_min": 0, "x_max": 2, "y_min": -1, "y_max": 1},
            "target": {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 1},
        },
    )
    r = _recompile(body)
    assert r.status_code == 409, r.text
    data = r.json()
    assert data["error"] == "no_feasible_plan"
    assert data["reason"] == "no_safe_path"
    assert data["last_reachable_slot"] == 4
    assert data["reachable_states"] == 1


def test_409_differs_from_422_for_bad_prefix():
    """前缀编号不合法是 422；前缀合法但无解才是 409。"""
    bad_prefix = _body(executed=[99])
    assert _recompile(bad_prefix).status_code == 422

    slots = [
        {
            "disturbance": [0, 0],
            "commands": [
                {"id": 1, "mode": "A", "correction": [0, 0], "energy": 1.0},
                {"id": 2, "mode": "B", "correction": [0, 0], "energy": 2.0},
            ],
        }
    ] * 4 + [
        {
            "disturbance": [1, 0],
            "commands": [
                {"id": 1, "mode": "A", "correction": [0, 0], "energy": 1.0},
                {"id": 2, "mode": "B", "correction": [1, 0], "energy": 1.0},
            ],
        }
    ] * 4
    no_suffix = _body(
        slots=slots,
        executed=[1, 1, 1, 1],
        observed=(2, 0),
        safety={"x_min": 0, "x_max": 2, "y_min": -1, "y_max": 1},
        target={"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 1},
    )
    assert _recompile(no_suffix).status_code == 409
