import sys
from collections import deque
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

# Define the stages of the pipeline
class Stage(Enum):
    PENDING = 0
    TEXT_GENERATION = 1    # LLM (Thinking/Prompting)
    IMAGE_GENERATION = 2   # Diffusion + VAE
    DONE = 3

@dataclass
class TaskState:
    """
    Holds the state and context for a single task as it moves through the pipeline.
    """
    task_spec: Any  # The original TaskSpec object
    stage: Stage = Stage.PENDING
    
    # We maintain a unique generation context for each task to prevent
    # KV-cache collisions between tasks running in parallel streams.
    gen_context: Optional[Dict[str, Any]] = None
    
    # Store intermediate results
    generated_text: Optional[str] = None
    generated_image: Any = None # PIL Image
    
    # Configuration inputs for generation
    cfg_text_context: Optional[Dict[str, Any]] = None
    cfg_img_context: Optional[Dict[str, Any]] = None
    image_shape: tuple = (1024, 1024)

class PipelineScheduler:
    def __init__(self):
        # Queues for each processing stage
        self.pending_queue = deque()
        self.text_queue = deque()
        self.image_queue = deque()
        self.finished_tasks = []

    def add_task(self, task_spec):
        """Register a new task."""
        state = TaskState(task_spec=task_spec)
        self.pending_queue.append(state)

    def get_next_text_task(self) -> Optional[TaskState]:
        """Fetch a task ready for LLM Text Generation."""
        # If we have tasks waiting to start, move them to text queue
        if self.pending_queue:
            task = self.pending_queue.popleft()
            task.stage = Stage.TEXT_GENERATION
            return task
        return None

    def get_next_image_task(self) -> Optional[TaskState]:
        """Fetch a task ready for Diffusion Image Generation."""
        if self.image_queue:
            task = self.image_queue.popleft()
            task.stage = Stage.IMAGE_GENERATION
            return task
        return None

    def submit_text_result(self, task: TaskState, result_text: str, next_contexts: dict):
        """
        Mark text generation as complete.
        Prepare the task for the image generation queue.
        """
        task.generated_text = result_text
        # Update context with the generated text (simulating the 'thought' process)
        task.gen_context = next_contexts['gen_context']
        task.cfg_text_context = next_contexts['cfg_text_context']
        task.cfg_img_context = next_contexts['cfg_img_context']
        
        # Move to next stage
        self.image_queue.append(task)

    def submit_image_result(self, task: TaskState, result_image):
        """Mark image generation as complete."""
        task.generated_image = result_image
        task.stage = Stage.DONE
        self.finished_tasks.append(task)

    def has_pending_work(self) -> bool:
        """Check if any queues still have active work."""
        return (len(self.pending_queue) > 0 or 
                len(self.text_queue) > 0 or 
                len(self.image_queue) > 0)