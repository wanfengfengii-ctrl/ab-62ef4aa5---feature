"""整窗动量计划动态规划（含余窗重编）。

每个时隙恰好选择一条候选指令，状态演化：

    p_{t+1} = p_t + disturbance_t + correction_t

要求 p_1 ... p_T 全部位于统一矩形安全域，且 p_T 位于目标矩形。

优化目标按严格优先级依次最小化：
  1. 总能耗；
  2. 相邻时隙模式切换次数；
  3. 指令编号序列（按字典序）。

实现上以前向动态规划逐层扩展可达前沿。分桶键为 ``(动量, 上一条指令模式)``：
因为模式切换次数依赖“上一条指令的模式”，同一动量、不同末模式的两条路径
不能简单互相支配，必须分桶保存；桶内未来代价只取决于桶键，保留字典序最小
标签即为安全支配。

余窗重编（:func:`recompile_plan`）固定已执行前缀，从最新遥测动量继续同一
DP：根节点继承前缀的能耗、切换次数、编号序列与末模式，因此前缀内部与
前后缀交界的模式切换都会计入整窗目标。
"""

from dataclasses import dataclass
from typing import Optional

from .models import Command, CompileRequest, Rectangle

Vec = tuple[int, int]
BucketKey = tuple[Vec, Optional[str]]

# 能耗来自 JSON 浮点，不同指令组合求和可能产生 1e-16 级误差，
# 该容差内视为能耗并列，交由切换次数与编号序列裁决。
ENERGY_EPS = 1e-9

# 标签类型：(总能耗, 切换次数, 编号序列)
Label = tuple[float, int, tuple[int, ...]]


def rank_lt(a: Label, b: Label) -> bool:
    """三级目标严格优先级下标签 a 是否优于 b。"""
    if a[0] < b[0] - ENERGY_EPS:
        return True
    if a[0] > b[0] + ENERGY_EPS:
        return False
    if a[1] != b[1]:
        return a[1] < b[1]
    return a[2] < b[2]


class InfeasiblePlanError(Exception):
    """不存在满足约束的计划。

    ``last_reachable_slot`` 为仍存在可达状态的最后时隙编号（已完成的时隙数，
    0 表示第一时隙就无法留在安全域）；``reachable_states`` 为该层不同可达
    动量状态的数量。
    """

    def __init__(self, reason: str, last_reachable_slot: int, reachable_states: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.last_reachable_slot = last_reachable_slot
        self.reachable_states = reachable_states


@dataclass(frozen=True)
class _Node:
    """一条到达某桶键的最优部分计划标签。"""

    energy: float
    switches: int
    ids: tuple[int, ...]
    mode: Optional[str]
    prev_key: Optional[BucketKey]
    command: Optional[Command]

    def label(self) -> Label:
        return (self.energy, self.switches, self.ids)


def _in_rect(p: Vec, rect: Rectangle) -> bool:
    return (
        rect.x_min <= p[0] <= rect.x_max
        and rect.y_min <= p[1] <= rect.y_max
    )


def _distinct_states(layer: dict[BucketKey, _Node]) -> int:
    return len({state for state, _mode in layer})


def _expand_layer(
    previous: dict[BucketKey, _Node], slot, safety: Rectangle
) -> dict[BucketKey, _Node]:
    """从上一层前沿扩展一个时隙，返回新的可达前沿。"""
    current: dict[BucketKey, _Node] = {}
    for (state, prev_mode), node in previous.items():
        for cmd in slot.commands:
            cx, cy = cmd.correction
            dx, dy = slot.disturbance
            nxt_state: Vec = (state[0] + dx + cx, state[1] + dy + cy)
            if not _in_rect(nxt_state, safety):
                continue

            switches = node.switches
            if prev_mode is not None and prev_mode != cmd.mode:
                switches += 1

            candidate = _Node(
                energy=node.energy + cmd.energy,
                switches=switches,
                ids=node.ids + (cmd.id,),
                mode=cmd.mode,
                prev_key=(state, prev_mode),
                command=cmd,
            )
            key: BucketKey = (nxt_state, cmd.mode)
            incumbent = current.get(key)
            if incumbent is None or rank_lt(candidate.label(), incumbent.label()):
                current[key] = candidate
    return current


def _build_response(
    request: CompileRequest,
    layers: list[dict[BucketKey, _Node]],
    best_key: BucketKey,
    extra: Optional[dict] = None,
) -> dict:
    """从终层最优桶回溯，组装与原计划同构的响应体。"""
    final_layer = layers[-1]
    total_slots = len(request.slots)

    selected_commands: list[Command] = []
    momentums: list[Vec] = []
    key: Optional[BucketKey] = best_key
    for t in range(total_slots, 0, -1):
        node = layers[t][key]
        # 固定前缀层的节点不参与回溯（layers[0:k+1] 中前缀节点没有 command），
        # 一旦进入无根 command 的层即停止。
        if node.command is None:
            break
        selected_commands.append(node.command)
        momentums.append(key[0])  # type: ignore[index]
        key = node.prev_key
    selected_commands.reverse()
    momentums.reverse()

    replanned = len(selected_commands)
    first_slot = total_slots - replanned + 1
    selected = [
        {
            "slot": idx,
            "command_id": cmd.id,
            "mode": cmd.mode,
            "correction": tuple(cmd.correction),
            "energy": cmd.energy,
        }
        for idx, cmd in enumerate(selected_commands, start=first_slot)
    ]

    root = final_layer[best_key]
    result = {
        "feasible": True,
        "slot_count": total_slots,
        "selected": selected,
        "momentums": momentums,
        "final_momentum": momentums[-1],
        "objectives": {
            "total_energy": root.energy,
            "mode_switches": root.switches,
            "command_id_sequence": list(root.ids),
        },
    }
    if extra:
        result.update(extra)
    return result


def _solve(
    request: CompileRequest,
    start_layer: dict[BucketKey, _Node],
    start_index: int,
    extra: Optional[dict] = None,
) -> dict:
    """从给定前沿（已完成 ``start_index`` 个时隙）跑到末层并选终态最优。

    无解时抛出 :class:`InfeasiblePlanError`，时隙编号沿用原窗口。
    """
    safety = request.safety_region
    target = request.target_region

    layers: list[Optional[dict[BucketKey, _Node]]] = [None] * start_index
    layers.append(start_layer)
    for t in range(start_index + 1, len(request.slots) + 1):
        previous = layers[t - 1]
        assert previous is not None
        current = _expand_layer(previous, request.slots[t - 1], safety)
        if not current:
            # 安全域在本时隙被切断：上一层即最后可达层。
            raise InfeasiblePlanError(
                reason="no_safe_path",
                last_reachable_slot=t - 1,
                reachable_states=_distinct_states(previous),
            )
        layers.append(current)

    final_layer = layers[-1]
    assert final_layer is not None
    terminal = [key for key in final_layer if _in_rect(key[0], target)]
    if not terminal:
        raise InfeasiblePlanError(
            reason="target_unreachable",
            last_reachable_slot=len(request.slots),
            reachable_states=_distinct_states(final_layer),
        )

    best_key = terminal[0]
    for key in terminal[1:]:
        if rank_lt(final_layer[key].label(), final_layer[best_key].label()):
            best_key = key

    return _build_response(request, layers, best_key, extra)  # type: ignore[arg-type]


def compile_plan(request: CompileRequest) -> dict:
    """编译整窗计划，返回可直接序列化的响应体。

    无解时抛出 :class:`InfeasiblePlanError`。
    """

    initial: Vec = tuple(request.initial_momentum)  # type: ignore[assignment]

    # layer 0 只有初始状态、无历史模式。
    start_layer: dict[BucketKey, _Node] = {
        (initial, None): _Node(0.0, 0, (), None, None, None)
    }
    return _solve(request, start_layer, start_index=0)


def recompile_plan(
    request: CompileRequest,
    prefix_commands: list[Command],
    observed: Vec,
    replanned_from_slot: int,
) -> dict:
    """固定已执行前缀，从观测动量为剩余时隙重编计划。

    ``prefix_commands`` 为时隙 1..k-1 已执行的指令（共 ``replanned_from_slot-1``
    条），``observed`` 为执行完前缀后的遥测动量（即第 k 时隙的起始状态）。
    前缀的能耗、切换次数、编号序列与末模式被根节点继承，因此整窗目标覆盖
    完整窗口，前后缀交界的模式切换也计入。
    """
    energy = 0.0
    switches = 0
    ids: list[int] = []
    prev_mode: Optional[str] = None
    for cmd in prefix_commands:
        energy += cmd.energy
        if prev_mode is not None and prev_mode != cmd.mode:
            switches += 1
        prev_mode = cmd.mode
        ids.append(cmd.id)

    root = _Node(
        energy=energy,
        switches=switches,
        ids=tuple(ids),
        mode=prev_mode,
        prev_key=None,
        command=None,
    )
    start_layer: dict[BucketKey, _Node] = {(observed, prev_mode): root}
    return _solve(
        request,
        start_layer,
        start_index=replanned_from_slot - 1,
        extra={
            "replanned_from_slot": replanned_from_slot,
            "observed_momentum": list(observed),
        },
    )
