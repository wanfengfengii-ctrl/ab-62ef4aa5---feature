"""FastAPI 应用入口。"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .models import CompileRequest, CompileResponse, RecompileRequest, RecompileResponse
from .planner import InfeasiblePlanError, compile_plan, recompile_plan
from .validation import SemanticError, validate_recompile, validate_semantics

app = FastAPI(
    title="动量卸载整窗计划编译服务",
    version="1.0.0",
)


@app.exception_handler(SemanticError)
async def _semantic_error_handler(_request: Request, exc: SemanticError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "invalid_request",
            "message": exc.message,
            "field": exc.field,
        },
    )


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    # 定位到具体字段（如 slots[3].commands[0].energy），方便值班员区分参数错误。
    details = []
    for err in exc.errors():
        loc = [str(part) for part in err["loc"] if part != "body"]
        details.append(
            {
                "field": ".".join(loc) if loc else "body",
                "type": err["type"],
                "message": err["msg"],
            }
        )
    return JSONResponse(
        status_code=422,
        content={"error": "invalid_request", "details": details},
    )


@app.exception_handler(InfeasiblePlanError)
async def _infeasible_handler(
    _request: Request, exc: InfeasiblePlanError
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "error": "no_feasible_plan",
            "reason": exc.reason,
            "message": "不存在满足安全域与末态目标的完整指令序列",
            "last_reachable_slot": exc.last_reachable_slot,
            "reachable_states": exc.reachable_states,
        },
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/momentum-plans/compile", response_model=CompileResponse)
async def compile_momentum_plan(request: CompileRequest) -> dict:
    validate_semantics(request)
    return compile_plan(request)


@app.post("/api/momentum-plans/recompile", response_model=RecompileResponse)
async def recompile_momentum_plan(request: RecompileRequest) -> dict:
    validate_recompile(request)

    original = request.original_request
    ids = request.executed_command_ids
    k = len(ids)

    # 校验已保证每个编号属于对应时隙；取回已执行指令对象（保留模式与能耗）。
    prefix_commands = []
    for i, cid in enumerate(ids):
        cmd = next(c for c in original.slots[i].commands if c.id == cid)
        prefix_commands.append(cmd)

    observed = tuple(request.observed_momentum)
    return recompile_plan(
        request=original,
        prefix_commands=prefix_commands,
        observed=observed,  # type: ignore[arg-type]
        replanned_from_slot=k + 1,
    )
