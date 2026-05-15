"""Pipeline state machine. Stages must advance in order."""

from enum import IntEnum


class PipelineStage(IntEnum):
    INIT = 0
    INPUTS_LOADED = 1
    TEXT_PREPROCESSED = 2
    MODEL_PROMPTED = 3
    STRUCTURED_OUTPUT_PARSED = 4
    CONFIDENCE_CHECKED = 5
    ROUTED = 6
    RESPONSE_GENERATED = 7
    RESULTS_SAVED = 8
    EVALUATION_COMPUTED = 9
    VALIDATION_COMPLETED = 10


class StageTracker:
    def __init__(self) -> None:
        self.current = PipelineStage.INIT
        self.history: list[tuple[PipelineStage, str]] = [(self.current, "pipeline initialized")]

    def advance(self, target: PipelineStage, note: str = "") -> None:
        if target <= self.current:
            raise RuntimeError(
                f"Cannot advance to {target.name}: current stage is {self.current.name}"
            )
        if target != self.current + 1:
            raise RuntimeError(
                f"Cannot skip stages: {self.current.name} -> {target.name} "
                f"(expected {PipelineStage(self.current + 1).name})"
            )
        self.current = target
        self.history.append((target, note))

    def require(self, stage: PipelineStage) -> None:
        if self.current < stage:
            raise RuntimeError(
                f"Stage {stage.name} required but current is {self.current.name}"
            )
