from trainer.scheduler import OptimizationStage, OptimizationDirective

DEFAULT_OPTIMIZATION_STAGES = [
    OptimizationStage(
        name="init",
        start=0,
        end=1000,
        directive=OptimizationDirective(
            stage_id=0,
            train_deform=False,
            train_static=False,
            train_dynamic=True
        )
    ),
    OptimizationStage(
        name="dynamic",
        start=1000,
        end=5000,
        directive=OptimizationDirective(
            stage_id=1,
            train_deform=True,
            train_static=False,
            train_dynamic=True
        )
    ),
    OptimizationStage(
        name="full",
        start=5000,
        end=40000,
        directive=OptimizationDirective(
            stage_id=2,
            train_deform=True,
            train_static=True,
            train_dynamic=True
        )
    )
]
