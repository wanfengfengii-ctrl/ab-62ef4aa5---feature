"""请求/响应数据模型。

动量、扰动、修正量统一使用二维整数向量（JSON 数组，长度恰好为 2）。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

IntVec = tuple[int, int]


class StrictModel(BaseModel):
    """禁止多余字段，字段拼写错误会被精确定位为参数错误。"""

    model_config = ConfigDict(extra="forbid")


class Command(StrictModel):
    """单个时隙内的一条候选磁力矩指令。"""

    id: int = Field(description="候选指令编号，在同一时隙内唯一")
    mode: str = Field(min_length=1, description="磁力矩器工作模式")
    correction: IntVec = Field(description="二维修正量")
    energy: float = Field(ge=0.0, description="非负能耗")

    @field_validator("energy")
    @classmethod
    def _energy_must_be_finite(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("能耗必须是有限实数")
        return value


class Slot(StrictModel):
    """一个卸载时隙：扰动加候选指令。"""

    disturbance: IntVec
    commands: list[Command] = Field(min_length=2, max_length=5)


class Rectangle(StrictModel):
    """统一矩形区域（含边界）。"""

    x_min: int
    x_max: int
    y_min: int
    y_max: int


class CompileRequest(StrictModel):
    """整窗计划编译请求。"""

    initial_momentum: IntVec
    safety_region: Rectangle
    target_region: Rectangle
    slots: list[Slot] = Field(min_length=8, max_length=20)


class SelectedCommand(StrictModel):
    slot: int
    command_id: int
    mode: str
    correction: IntVec
    energy: float


class Objectives(StrictModel):
    total_energy: float
    mode_switches: int
    command_id_sequence: list[int]


class CompileResponse(StrictModel):
    feasible: Literal[True]
    slot_count: int
    selected: list[SelectedCommand]
    momentums: list[IntVec]
    final_momentum: IntVec
    objectives: Objectives


class RecompileRequest(StrictModel):
    """在轨余窗重编请求。

    携带原编译请求、已执行前缀的指令编号（按时隙顺序）与最新遥测观测动量。
    前缀长度的跨字段上限（不超过总时隙数减一）由语义校验补充。
    """

    request: CompileRequest
    executed_command_ids: list[int] = Field(min_length=1)
    observed_momentum: IntVec


class RecompileResponse(CompileResponse):
    """余窗重编响应：沿用原计划结构，另标明重编起点时隙与观测动量。"""

    # 第一条重新选择指令所在的时隙编号（已执行前缀长度 + 1）。
    replanned_from_slot: int
    observed_momentum: IntVec
