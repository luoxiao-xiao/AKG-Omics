from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class TaskSpec:
    task_id: str
    source_modalities: List[str]
    target_modality: str
    species: str = "human"
    tissue: Optional[str] = None
    required_relations: List[str] = field(default_factory=list)

    def normalized(self) -> "TaskSpec":
        src = [str(m).strip().lower() for m in self.source_modalities]
        tgt = str(self.target_modality).strip().lower()
        req = [str(r).strip().lower() for r in self.required_relations]
        return TaskSpec(
            task_id=str(self.task_id),
            source_modalities=src,
            target_modality=tgt,
            species=str(self.species).strip().lower(),
            tissue=None if self.tissue is None else str(self.tissue).strip().lower(),
            required_relations=req,
        )

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SourceCandidate:
    source_id: str
    score: float
    is_mandatory: bool
    is_recommended: bool
    available: bool
    reasons: List[str] = field(default_factory=list)
    data_coverage_score: Optional[float] = None
    data_coverage_detail: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SelectionResult:
    mode: str
    selected_source_ids: List[str]
    candidates: List[SourceCandidate]
    notes: List[str] = field(default_factory=list)
    agent_decision: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "mode": self.mode,
            "selected_source_ids": self.selected_source_ids,
            "candidates": [c.to_dict() for c in self.candidates],
            "notes": self.notes,
            "agent_decision": self.agent_decision,
        }
