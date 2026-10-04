"""整窗动量计划动态规划，含在轨余窗重编。

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

在轨重编时已执行前缀固定，动态规划只在剩余时隙上从遥测观测动量继续：
前缀的能耗、切换次数（含前后缀交界）与编号作为初始标签累加，末态仍须进入
原目标域，输出覆盖完整窗口。
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

    余窗重编下二者以**原窗口时隙编号**报告：前缀时隙按其原编号计入，
    因此重编起点为 ``k+1``，立即切断时最后可达时隙为 ``k``。
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


@dataclass(frozen=True)
class _Prefix:
    """重编时固定的已执行前缀标签。

    ``mode`` 为前缀最后一条指令的模式（决定前后缀交界是否计切换）；
    重编起点为第 ``k+1`` 个时隙，``layers`` 只包含剩余时隙的扩展结果，
    其第 0 层即此前缀层。
    """

    k: int
    energy: float
    switches: int
    ids: tuple[int, ...]
    mode: str


def _run_dp(
    request: CompileRequest,
    start_state: Vec,
    prefix: Optional[_Prefix],
) -> dict:
    """在请求时隙上运行前向动态规划。

    ``prefix`` 为空时整窗编译：从初始动量、第 1 时隙开始，标签从零累计；
    非空时余窗重编：``layers[0]`` 固定为已执行前缀层（观测动量、前缀模式
    与累计标签），只扩展第 ``k+1 … T`` 个时隙。
    """

    safety = request.safety_region
    target = request.target_region

    if prefix is None:
        first_slot = 1
        start_key: BucketKey = (start_state, None)
        layers: list[dict[BucketKey, _Node]] = [
            {start_key: _Node(0.0, 0, (), None, None, None)}
        ]
    else:
        first_slot = prefix.k + 1
        start_key = (start_state, prefix.mode)
        layers = [
            {
                start_key: _Node(
                    prefix.energy,
                    prefix.switches,
                    prefix.ids,
                    prefix.mode,
                    None,
                    None,
                )
            }
        ]

    for t in range(first_slot, len(request.slots) + 1):
        slot = request.slots[t - 1]
        previous = layers[-1]
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

        if not current:
            # 安全域在本时隙被切断：上一层即最后可达层。
            # 层数偏移即前缀长度，报告原窗口时隙编号。
            raise InfeasiblePlanError(
                reason="no_safe_path",
                last_reachable_slot=t - 1,
                reachable_states=_distinct_states(previous),
            )
        layers.append(current)

    final_layer = layers[-1]
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

    # 回溯重建逐隙选择与动量（仅本次扩展的时隙；前缀已固定）。
    selected_commands: list[Command] = []
    momentums: list[Vec] = []
    key: Optional[BucketKey] = best_key
    for t in range(len(request.slots), first_slot - 1, -1):
        node = layers[t - first_slot + 1][key]
        assert node.command is not None and node.prev_key is not None
        selected_commands.append(node.command)
        momentums.append(key[0])  # type: ignore[index]
        key = node.prev_key
    selected_commands.reverse()
    momentums.reverse()

    root = final_layer[best_key]
    selected = [
        {
            "slot": idx,
            "command_id": cmd.id,
            "mode": cmd.mode,
            "correction": tuple(cmd.correction),
            "energy": cmd.energy,
        }
        for idx, cmd in enumerate(
            selected_commands,
            start=1 if prefix is None else first_slot,
        )
    ]

    return {
        "feasible": True,
        "slot_count": len(request.slots),
        "selected": selected,
        "momentums": momentums,
        "final_momentum": momentums[-1],
        "objectives": {
            "total_energy": root.energy,
            "mode_switches": root.switches,
            "command_id_sequence": list(root.ids),
        },
    }


def compile_plan(request: CompileRequest) -> dict:
    """编译整窗计划，返回可直接序列化的响应体。

    无解时抛出 :class:`InfeasiblePlanError`。
    """

    initial: Vec = tuple(request.initial_momentum)  # type: ignore[assignment]
    return _run_dp(request, initial, None)


def recompile_plan(
    request: CompileRequest,
    executed_command_ids: list[int],
    observed_momentum: Vec,
) -> dict:
    """固定已执行前缀，仅为剩余时隙重新选择指令。

    前缀编号的合法性（长度 1 … T-1、各编号依次属于对应时隙、无缺失）
    与观测动量位于安全域由语义校验先行保证；本函数只负责：

    1. 按原模型演化重放前缀，累计前缀能耗、模式与切换次数（模式只取决于
       已执行编号，与观测偏差无关）；
    2. 从观测动量起在剩余时隙上运行动态规划，末态仍进入原目标域；
    3. 拼接前缀，输出覆盖完整窗口的编号序列、总能耗与切换次数。

    无安全后缀时抛出 :class:`InfeasiblePlanError`，时隙编号按原窗口报告。
    """

    k = len(executed_command_ids)

    # 重放已执行前缀：定位编号对应的候选指令，累计标签。
    prefix_energy = 0.0
    prefix_switches = 0
    prefix_ids: tuple[int, ...] = ()
    prefix_commands: list[Command] = []
    prev_mode: Optional[str] = None
    for t in range(k):
        cmd = next(
            c for c in request.slots[t].commands if c.id == executed_command_ids[t]
        )
        prefix_energy += cmd.energy
        if prev_mode is not None and prev_mode != cmd.mode:
            prefix_switches += 1
        prev_mode = cmd.mode
        prefix_ids += (cmd.id,)
        prefix_commands.append(cmd)
    assert prev_mode is not None  # k >= 1

    result = _run_dp(
        request,
        observed_momentum,
        _Prefix(
            k=k,
            energy=prefix_energy,
            switches=prefix_switches,
            ids=prefix_ids,
            mode=prev_mode,
        ),
    )

    # 拼接已执行前缀。前 k-1 隙动量按原模型从初始动量重放（计划档轨迹）；
    # 第 k 隙末态即实测观测动量——实测偏差正发生在此处，后缀也从该点继续，
    # 因此交界之后的演化连续，且上报的每一隙动量都位于安全域。
    prefix_selected = [
        {
            "slot": idx,
            "command_id": cmd.id,
            "mode": cmd.mode,
            "correction": tuple(cmd.correction),
            "energy": cmd.energy,
        }
        for idx, cmd in enumerate(prefix_commands, start=1)
    ]
    prefix_momentums: list[Vec] = []
    state: Vec = tuple(request.initial_momentum)  # type: ignore[assignment]
    for t in range(k - 1):
        d = request.slots[t].disturbance
        cx, cy = prefix_commands[t].correction
        state = (state[0] + d[0] + cx, state[1] + d[1] + cy)
        prefix_momentums.append(state)
    prefix_momentums.append(observed_momentum)

    result["selected"] = prefix_selected + result["selected"]
    result["momentums"] = prefix_momentums + result["momentums"]
    result["replanned_from_slot"] = k + 1
    result["observed_momentum"] = list(observed_momentum)
    return result
