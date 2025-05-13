from typing import List, NamedTuple


class OptimizationDirective(NamedTuple):
    stage_id: int
    train_deform: bool
    train_static: bool
    train_dynamic: bool


class OptimizationStage:
    def __init__(self, name: str, start: int, end: int, directive: OptimizationDirective):
        assert start < end, f"Invalid stage bounds: {start} >= {end}"
        self.name = name
        self.start = start
        self.end = end
        self.directive = directive

    @property
    def duration(self) -> int:
        return self.end - self.start

    def contains(self, iteration: int) -> bool:
        return self.start <= iteration < self.end

    def __repr__(self) -> str:
        return f"<Stage '{self.name}' [{self.start}, {self.end})>"


class OptimizationStageScheduler:
    def __init__(self, stages: List[OptimizationStage]):
        assert stages, "Scheduler must be initialized with at least one stage"
        self._stages = stages

    def directive_for(self, iteration: int):
        return self.get_stage_at_iter(iteration).directive

    def get_stage_at_iter(self, iteration: int):
        for stage in self._stages:
            if stage.contains(iteration):
                return stage
        return self._stages[-1]
