from dataclasses import dataclass

@dataclass
class Task:
    id: int
    title: str
    completed: bool

    def __repr__(self):
        return f"Task({self.id}, '{self.title}', {self.completed})"

    def __eq__(self, other):
        if not isinstance(other, Task):
            return False
        return (self.id == other.id) and (self.title == other.title) and (self.completed == other.completed)