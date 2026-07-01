from dataclasses import dataclass

@dataclass
class Task:
    id: int
    title: str
    done: bool