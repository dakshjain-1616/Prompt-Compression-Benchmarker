from pcb.tasks.base import BaseTask, TaskResult
from pcb.tasks.rag import RAGTask
from pcb.tasks.summarization import SummarizationTask
from pcb.tasks.coding import CodingTask

ALL_TASKS = {
    "rag": RAGTask,
    "summarization": SummarizationTask,
    "coding": CodingTask,
}

__all__ = ["BaseTask", "TaskResult", "RAGTask", "SummarizationTask", "CodingTask", "ALL_TASKS"]
