"""一次性核验脚本（compose 服务 verify）。

顺序执行并以退出码报告：
  1. 等待并确认 API 健康；
  2. 核对代码测试（pytest 全套）；
  3. 核对镜像构建契约（非 root、应用与工具链可导入、健康端点契约）；
  4. 业务冒烟：可行计划、并列裁决（三级目标）、真实无解诊断、
     参数错误与无解的区分，以及余窗重编（前缀固定、观测续演、
     422 前缀/观测非法与 409 无安全后缀的区分）。
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

# 以脚本方式运行时，把项目根（app 包所在目录）加入 sys.path。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

API_BASE = os.environ.get("API_BASE_URL", "http://api:8000")
COMPILE_URL = f"{API_BASE}/api/momentum-plans/compile"
RECOMPILE_URL = f"{API_BASE}/api/momentum-plans/recompile"
HEALTH_URL = f"{API_BASE}/health"


def _log(ok: bool, message: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {message}")


def _post_json(url: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _wait_for_health(timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception as exc:  # noqa: BLE001 - 启动中任何异常都重试
            last_err = exc
            time.sleep(1.0)
    print(f"等待 API 健康超时：{last_err}")
    return False


def _slot(disturbance, *commands):
    return {
        "disturbance": list(disturbance),
        "commands": [
            {"id": cid, "mode": mode, "correction": list(corr), "energy": energy}
            for cid, mode, corr, energy in commands
        ],
    }


WIDE_SAFETY = {"x_min": -100, "x_max": 100, "y_min": -100, "y_max": 100}


def _base_body(slots, safety=None, target=None, initial=(0, 0)):
    return {
        "initial_momentum": list(initial),
        "safety_region": safety or dict(WIDE_SAFETY),
        "target_region": target or dict(WIDE_SAFETY),
        "slots": slots,
    }


def smoke_feasible() -> bool:
    """可行计划：返回所选指令、逐隙动量与三项目标值。"""
    slots = [
        _slot((0, 0), (1, "A", (1, 0), 1.0), (2, "B", (-1, 0), 2.0)),
        _slot((0, 1), (1, "A", (0, -1), 1.0), (2, "B", (0, 0), 1.0)),
        _slot((1, 0), (1, "A", (-1, 0), 1.0), (2, "B", (0, 1), 3.0)),
        _slot((0, 0), (1, "A", (0, 0), 2.0), (2, "B", (0, 0), 1.0)),
        _slot((-1, 0), (1, "A", (1, 0), 1.0), (2, "B", (0, 0), 2.0)),
        _slot((0, -1), (1, "A", (0, 1), 1.0), (2, "B", (0, 0), 1.5)),
        _slot((0, 0), (1, "A", (0, 0), 1.0), (2, "B", (1, 0), 0.5)),
        _slot((0, 0), (1, "A", (0, 0), 1.0), (2, "B", (0, -1), 2.0)),
    ]
    status, data = _post_json(COMPILE_URL, _base_body(slots))
    if status != 200 or not data.get("feasible"):
        _log(False, f"可行计划应返回 200/feasible=true，实际 {status}: {data}")
        return False

    ok = True
    if len(data["selected"]) != 8 or len(data["momentums"]) != 8:
        ok = False
    obj = data["objectives"]
    for key in ("total_energy", "mode_switches", "command_id_sequence"):
        if key not in obj:
            ok = False

    # 连续演化复核：前态 + 扰动 + 修正量 == 逐隙动量，且均在安全域。
    state = (0, 0)
    for i, sel in enumerate(data["selected"]):
        d = slots[i]["disturbance"]
        c = sel["correction"]
        state = (state[0] + d[0] + c[0], state[1] + d[1] + c[1])
        if tuple(data["momentums"][i]) != state:
            ok = False
    if tuple(data["final_momentum"]) != tuple(data["momentums"][-1]):
        ok = False

    _log(ok, "业务冒烟 1/5：可行计划（选择、逐隙动量、三项目标）")
    return ok


def smoke_tie_breaking() -> bool:
    """并列裁决：能耗相同优先少切换，再并列取编号序列字典序最小。"""
    # 每时隙两条同模式指令、修正相同、能耗相同：id=3 字典序更优、零切换。
    slots = [
        _slot((0, 0), (7, "A", (0, 0), 1.0), (3, "A", (0, 0), 1.0))
    ] * 8
    status, data = _post_json(COMPILE_URL, _base_body(slots))
    ok = (
        status == 200
        and data["objectives"]["total_energy"] == 8.0
        and data["objectives"]["mode_switches"] == 0
        and data["objectives"]["command_id_sequence"] == [3] * 8
    )
    _log(ok, f"业务冒烟 2/5：并列裁决（能耗→切换数→编号序列） 实际目标={data.get('objectives') if status == 200 else data}")
    return ok


def smoke_infeasible() -> bool:
    """真实无解：409 且报告最后可达时隙与该层可达状态数。"""
    # 第一隙结果 x ∈ {+1,-1}，安全域只含原点 → 立即切断。
    slots = [
        _slot((0, 0), (1, "A", (1, 0), 1.0), (2, "B", (-1, 0), 1.0))
    ] * 8
    body = _base_body(
        slots,
        safety={"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0},
        target={"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0},
    )
    status, data = _post_json(COMPILE_URL, body)
    ok = (
        status == 409
        and data.get("error") == "no_feasible_plan"
        and data.get("last_reachable_slot") == 0
        and data.get("reachable_states") == 1
    )
    _log(
        ok,
        f"业务冒烟 3/5：真实无解诊断（409、最后可达时隙、可达状态数） 实际 {status}: {data if not ok else 'last_reachable_slot=0, reachable_states=1'}",
    )
    return ok


def smoke_param_error() -> bool:
    """参数矛盾：422 且定位字段，与 409 真实无解明确区分。"""
    slots = [
        _slot((0, 0), (1, "A", (1, 0), 1.0), (2, "B", (-1, 0), 1.0))
    ] * 8
    body = _base_body(slots)
    body["initial_momentum"] = [0, "bad"]
    status, data = _post_json(COMPILE_URL, body)
    ok = status == 422 and data.get("error") == "invalid_request"
    located = any(
        "initial_momentum" in d.get("field", "") for d in data.get("details", [])
    )
    _log(
        ok and located,
        f"业务冒烟 4/5：参数错误定位字段（422，区别于 409） 实际 {status}: {data}",
    )
    return ok and located


def smoke_recompile() -> bool:
    """余窗重编：固定已执行前缀、从观测动量续演，422/409 区分。"""
    ok = True
    detail = ""

    # 场景一：全程零扰动、两条同能耗同修正指令；前缀 [1,1,1]、观测仍在原点。
    # 后缀应按三级裁决继续选 id=1，整窗序列 [1]*8、零切换、能耗 8。
    slots = [
        _slot((0, 0), (1, "A", (0, 0), 1.0), (2, "B", (0, 0), 1.0))
    ] * 8
    body = _base_body(slots)
    status, data = _post_json(
        RECOMPILE_URL,
        {
            "request": body,
            "executed_command_ids": [1, 1, 1],
            "observed_momentum": [0, 0],
        },
    )
    if status != 200:
        ok = False
        detail = f"重编应返回 200，实际 {status}: {data}"
    else:
        obj = data["objectives"]
        checks = [
            data.get("replanned_from_slot") == 4,
            data.get("observed_momentum") == [0, 0],
            obj.get("command_id_sequence") == [1] * 8,
            obj.get("mode_switches") == 0,
            obj.get("total_energy") == 8.0,
            len(data.get("selected", [])) == 8,
            [s["command_id"] for s in data["selected"][:3]] == [1, 1, 1],
            data.get("momentums", [None] * 4)[2] == [0, 0],
        ]
        if not all(checks):
            ok = False
            detail = f"重编计划/目标字段不符：{data}"

        # 后缀连续演化复核：从观测动量起逐隙 p += 扰动 + 修正。
        state = (0, 0)
        for i in range(3, 8):
            sel = data["selected"][i]
            d = slots[i]["disturbance"]
            state = (
                state[0] + d[0] + sel["correction"][0],
                state[1] + d[1] + sel["correction"][1],
            )
            if list(state) != data["momentums"][i]:
                ok = False
                detail = f"后缀第 {i + 1} 隙动量未从观测点连续演化：{data}"

    # 场景二：前缀编号不存在 → 422 并定位字段。
    bad_status, bad_data = _post_json(
        RECOMPILE_URL,
        {
            "request": _base_body(slots),
            "executed_command_ids": [99],
            "observed_momentum": [0, 0],
        },
    )
    if not (
        bad_status == 422
        and bad_data.get("field") == "executed_command_ids[0]"
    ):
        ok = False
        detail = f"前缀编号缺失应 422 定位字段，实际 {bad_status}: {bad_data}"

    # 场景三：观测动量在安全域外 → 422。
    obs_status, obs_data = _post_json(
        RECOMPILE_URL,
        {
            "request": _base_body(slots),
            "executed_command_ids": [1],
            "observed_momentum": [101, 0],
        },
    )
    if obs_status != 422 or obs_data.get("field") != "observed_momentum":
        ok = False
        detail = f"观测动量越界应 422，实际 {obs_status}: {obs_data}"

    # 场景四：合法前缀与观测，但余窗第一步即被安全域切断 → 409，
    # last_reachable_slot 以原窗口编号报告（等于前缀长度 4）。
    cut_slots = [
        _slot((0, 0), (1, "A", (0, 0), 1.0), (2, "B", (0, 0), 2.0))
    ] * 4 + [
        _slot((1, 0), (1, "A", (0, 0), 1.0), (2, "B", (1, 0), 1.0))
    ] * 4
    cut_body = _base_body(
        cut_slots,
        safety={"x_min": 0, "x_max": 2, "y_min": -1, "y_max": 1},
        target={"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 1},
    )
    cut_status, cut_data = _post_json(
        RECOMPILE_URL,
        {
            "request": cut_body,
            "executed_command_ids": [1, 1, 1, 1],
            "observed_momentum": [2, 0],
        },
    )
    if not (
        cut_status == 409
        and cut_data.get("error") == "no_feasible_plan"
        and cut_data.get("last_reachable_slot") == 4
        and cut_data.get("reachable_states") == 1
    ):
        ok = False
        detail = f"无安全后缀应 409 并以原窗口编号报告，实际 {cut_status}: {cut_data}"

    _log(
        ok,
        f"业务冒烟 5/5：余窗重编（前缀固定、观测续演、422/409 区分） {detail}",
    )
    return ok


def image_contract() -> bool:
    """镜像构建契约核验。"""
    checks = [
        os.geteuid() != 0,  # 以非 root 运行
    ]
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        from app.main import app  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"应用导入失败：{exc}")
        return False
    _log(all(checks), f"镜像构建契约（非 root uid={os.geteuid()}、应用与工具链可导入）")
    return all(checks)


def run_pytest() -> bool:
    print("\n=== 运行代码测试 (pytest) ===")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
        cwd=_PROJECT_ROOT,
    )
    _log(proc.returncode == 0, f"代码测试（退出码 {proc.returncode}）")
    return proc.returncode == 0


def main() -> int:
    print("=== 等待 API 健康 ===")
    if not _wait_for_health():
        _log(False, "API 健康检查")
        return 1
    _log(True, "API 健康检查")

    results = [
        image_contract(),
        run_pytest(),
        smoke_feasible(),
        smoke_tie_breaking(),
        smoke_infeasible(),
        smoke_param_error(),
        smoke_recompile(),
    ]
    print("\n=== 核验汇总 ===")
    print(f"通过 {sum(results)}/{len(results)} 项")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
