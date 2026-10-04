"""请求语义校验：把“输入矛盾”精确定位到字段。

Pydantic 负责类型与基数（8–20 时隙、每时隙 2–5 条指令、二维整数向量、
非负能耗等），本模块补充跨字段约束：矩形边界顺序、目标域包含于安全域、
初始动量位于安全域、时隙内指令编号唯一。

余窗重编额外校验：已执行编号依次属于前 k 个时隙（1 ≤ k ≤ T-1，无重复），
观测动量位于安全域。
"""

from .models import CompileRequest, RecompileRequest


class SemanticError(Exception):
    def __init__(self, field: str, message: str) -> None:
        super().__init__(message)
        self.field = field
        self.message = message


def _rect_bounds_ok(rect) -> bool:
    return rect.x_min <= rect.x_max and rect.y_min <= rect.y_max


def validate_semantics(req: CompileRequest) -> None:
    safety = req.safety_region
    target = req.target_region

    if not _rect_bounds_ok(safety):
        raise SemanticError(
            "safety_region",
            "矩形边界矛盾：需要满足 x_min <= x_max 且 y_min <= y_max",
        )
    if not _rect_bounds_ok(target):
        raise SemanticError(
            "target_region",
            "矩形边界矛盾：需要满足 x_min <= x_max 且 y_min <= y_max",
        )

    # 末态必须同时位于安全域与目标域，因此目标域必须包含于安全域，
    # 否则属于输入自相矛盾，而非“真实无解”。
    if not (
        safety.x_min <= target.x_min
        and target.x_max <= safety.x_max
        and safety.y_min <= target.y_min
        and target.y_max <= safety.y_max
    ):
        raise SemanticError(
            "target_region",
            "目标矩形必须包含于统一安全域内，否则末态约束互相矛盾",
        )

    ix, iy = req.initial_momentum
    if not (
        safety.x_min <= ix <= safety.x_max and safety.y_min <= iy <= safety.y_max
    ):
        raise SemanticError(
            "initial_momentum",
            "初始动量必须位于安全域内",
        )

    for t, slot in enumerate(req.slots):
        seen: set[int] = set()
        for c, cmd in enumerate(slot.commands):
            if cmd.id in seen:
                raise SemanticError(
                    f"slots[{t}].commands[{c}].id",
                    f"时隙 {t}（0 基）内指令编号 {cmd.id} 重复，编号须在时隙内唯一",
                )
            seen.add(cmd.id)


def validate_recompile(req: RecompileRequest) -> None:
    """余窗重编语义校验，字段路径统一加 ``request.`` 前缀定位嵌套原请求。"""
    try:
        validate_semantics(req.request)
    except SemanticError as exc:
        raise SemanticError(f"request.{exc.field}", exc.message) from exc

    total = len(req.request.slots)
    executed = req.executed_command_ids
    k = len(executed)

    # 至少执行一个时隙才能称为“前缀”，且必须保留至少一个余窗时隙。
    # （空列表已由 Pydantic min_length 拦截，此处覆盖 k == total。）
    if k >= total:
        raise SemanticError(
            "executed_command_ids",
            f"已执行前缀须依次覆盖前 k 个时隙且 1 <= k <= {total - 1}，"
            f"当前 k={k}，没有可供重编的剩余时隙",
        )

    # 注意：指令编号仅要求“时隙内唯一”，同一编号在不同时隙重复出现完全合法
    # （例如整窗重复选择编号 1），因此不能把 executed 列表中的跨隙重号判错。
    # 前缀的每个位置只须对号入座：该编号必须就在对应时隙的候选集中。
    all_ids = {cmd.id for slot in req.request.slots for cmd in slot.commands}
    for t, cid in enumerate(executed):
        slot_ids = {cmd.id for cmd in req.request.slots[t].commands}
        if cid not in slot_ids:
            if cid not in all_ids:
                raise SemanticError(
                    f"executed_command_ids[{t}]",
                    f"已执行编号 {cid} 在整个窗口的候选指令中均不存在，"
                    f"第 {t + 1} 个时隙可选编号为 {sorted(slot_ids)}（前缀编号缺失）",
                )
            raise SemanticError(
                f"executed_command_ids[{t}]",
                f"已执行编号 {cid} 不属于第 {t + 1} 个时隙的候选指令，"
                f"前缀只能依次覆盖前 k 个时隙，该时隙可选编号为 {sorted(slot_ids)}",
            )

    safety = req.request.safety_region
    ox, oy = req.observed_momentum
    if not (
        safety.x_min <= ox <= safety.x_max and safety.y_min <= oy <= safety.y_max
    ):
        raise SemanticError(
            "observed_momentum",
            "观测动量必须位于统一安全域内，才能从该状态继续安全演化",
        )
