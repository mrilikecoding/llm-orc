from dataclasses import dataclass, asdict


@dataclass
class Task:
    id: int
    title: str
    done: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(**data)
