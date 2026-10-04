"""pytest 公共夹具与构造工具。"""

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_request(
    initial=(0, 0),
    safety=None,
    target=None,
    slots=None,
):
    """构造一个编译请求，默认 8 时隙、宽松安全域。"""
    if safety is None:
        safety = {"x_min": -100, "x_max": 100, "y_min": -100, "y_max": 100}
    if target is None:
        target = {"x_min": -100, "x_max": 100, "y_min": -100, "y_max": 100}
    return {
        "initial_momentum": list(initial),
        "safety_region": safety,
        "target_region": target,
        "slots": slots if slots is not None else [],
    }


def slot(disturbance, *commands):
    """commands: (id, mode, correction, energy) 元组序列。"""
    return {
        "disturbance": list(disturbance),
        "commands": [
            {
                "id": cid,
                "mode": mode,
                "correction": list(corr),
                "energy": energy,
            }
            for cid, mode, corr, energy in commands
        ],
    }


def brute_force_best(req: dict):
    """穷举所有指令组合，返回 (energy, switches, ids) 或 None。

    与生产 DP 的语义完全独立，作为对拍参考。
    """
    safety = req["safety_region"]
    target = req["target_region"]
    p = tuple(req["initial_momentum"])

    def in_rect(q, r):
        return r["x_min"] <= q[0] <= r["x_max"] and r["y_min"] <= q[1] <= r["y_max"]

    best = None
    for combo in itertools.product(*(s["commands"] for s in req["slots"])):
        state = p
        energy = 0.0
        switches = 0
        ids = []
        prev_mode = None
        ok = True
        for s, cmd in zip(req["slots"], combo):
            d = s["disturbance"]
            c = cmd["correction"]
            state = (state[0] + d[0] + c[0], state[1] + d[1] + c[1])
            if not in_rect(state, safety):
                ok = False
                break
            energy += cmd["energy"]
            if prev_mode is not None and prev_mode != cmd["mode"]:
                switches += 1
            prev_mode = cmd["mode"]
            ids.append(cmd["id"])
        if ok and in_rect(state, target):
            label = (energy, switches, tuple(ids))
            if best is None or label < best:
                best = label
    return best


def brute_force_recompile(req: dict, executed_ids, observed):
    """余窗重编的独立穷举参考：固定前缀代价，从观测动量穷举剩余时隙。

    前缀的能耗/切换/末模式折叠进根标签，前后缀交界的模式切换照常计入。
    返回 (energy, switches, ids) 或 None。
    """
    safety = req["safety_region"]
    target = req["target_region"]
    k = len(executed_ids)

    def in_rect(q, r):
        return r["x_min"] <= q[0] <= r["x_max"] and r["y_min"] <= q[1] <= r["y_max"]

    energy = 0.0
    switches = 0
    prev_mode = None
    for i, cid in enumerate(executed_ids):
        cmd = next(c for c in req["slots"][i]["commands"] if c["id"] == cid)
        energy += cmd["energy"]
        if prev_mode is not None and prev_mode != cmd["mode"]:
            switches += 1
        prev_mode = cmd["mode"]

    prefix_label = (energy, switches, prev_mode, list(executed_ids))
    suffix_slots = req["slots"][k:]
    best = None
    for combo in itertools.product(*(s["commands"] for s in suffix_slots)):
        state = tuple(observed)
        e, sw, pm, ids = prefix_label[0], prefix_label[1], prefix_label[2], list(prefix_label[3])
        ok = True
        for s, cmd in zip(suffix_slots, combo):
            d = s["disturbance"]
            c = cmd["correction"]
            state = (state[0] + d[0] + c[0], state[1] + d[1] + c[1])
            if not in_rect(state, safety):
                ok = False
                break
            e += cmd["energy"]
            if pm is not None and pm != cmd["mode"]:
                sw += 1
            pm = cmd["mode"]
            ids.append(cmd["id"])
        if ok and in_rect(state, target):
            label = (e, sw, tuple(ids))
            if best is None or label < best:
                best = label
    return best


def eight_slots():
    """固定 8 时隙基础场景：两个模式各两条指令。"""
    return [
        slot((0, 0), (1, "A", (1, 0), 1.0), (2, "B", (-1, 0), 2.0)),
        slot((0, 1), (1, "A", (0, -1), 1.0), (2, "B", (0, 0), 1.0)),
        slot((1, 0), (1, "A", (-1, 0), 1.0), (2, "B", (0, 1), 3.0)),
        slot((0, 0), (1, "A", (0, 0), 2.0), (2, "B", (0, 0), 1.0)),
        slot((-1, 0), (1, "A", (1, 0), 1.0), (2, "B", (0, 0), 2.0)),
        slot((0, -1), (1, "A", (0, 1), 1.0), (2, "B", (0, 0), 1.5)),
        slot((0, 0), (1, "A", (0, 0), 1.0), (2, "B", (1, 0), 0.5)),
        slot((0, 0), (1, "A", (0, 0), 1.0), (2, "B", (0, -1), 2.0)),
    ]
