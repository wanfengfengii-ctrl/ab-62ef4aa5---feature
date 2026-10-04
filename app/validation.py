"""请求语义校验：把“输入矛盾”精确定位到字段。

Pydantic 负责类型与基数（8–20 时隙、每时隙 2–5 条指令、二维整数向量、
非负能耗等），本模块补充跨字段约束：矩形边界顺序、目标域包含于安全域、
初始动量位于安全域、时隙内指令编号唯一。

余窗重编另外校验：已执行编号依次属于对应前缀时隙、前缀长度合法、
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
    """余窗重编的跨字段校验（原请求本身先按整窗编译规则校验）。"""
    original = req.original_request
    try:
        validate_semantics(original)
    except SemanticError as exc:
        # 原请求嵌在 original_request 字段下，错误定位加上字段前缀。
        raise SemanticError(f"original_request.{exc.field}", exc.message) from None

    slots = original.slots
    total = len(slots)
    ids = req.executed_command_ids
    k = len(ids)

    # 至少执行一个时隙、至多执行到倒数第二个时隙，剩余窗口不能为空。
    if k >= total:
        raise SemanticError(
            "executed_command_ids",
            f"已执行前缀占 {k} 个时隙，但总时隙数为 {total}：k 须满足 1 <= k <= 总时隙数-1",
        )

    # 编号只在时隙内唯一，跨时隙复用同一编号是正常情形（每时隙恰好一条），
    # 因此逐个位置核对：该编号必须存在于对应时隙的候选集合中。
    # 时隙内编号重复属于原请求自相矛盾，已由 validate_semantics 拦截。
    for i, cid in enumerate(ids):
        slot_ids = {cmd.id for cmd in slots[i].commands}
        if cid not in slot_ids:
            raise SemanticError(
                f"executed_command_ids[{i}]",
                f"编号 {cid} 不属于时隙 {i + 1} 的候选指令集合 {sorted(slot_ids)}",
            )

    ox, oy = req.observed_momentum
    safety = original.safety_region
    if not (
        safety.x_min <= ox <= safety.x_max and safety.y_min <= oy <= safety.y_max
    ):
        raise SemanticError(
            "observed_momentum",
            "观测动量必须位于安全域内，余窗安全演化无从继续",
        )
