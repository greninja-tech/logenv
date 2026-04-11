from pydantic import BaseModel
from typing import List, Optional


# ---------------- LOG STRUCTURES ----------------

class LogEntry(BaseModel):
    timestamp: str
    level: str
    service: str
    message: str


class SystemMetrics(BaseModel):
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    active_connections: int
    request_rate: float
    error_rate: float


class Alert(BaseModel):
    alert_id: str
    severity: str
    service: str
    message: str
    triggered_at: str


# ---------------- ENV INTERFACE ----------------

class Observation(BaseModel):
    logs: List[LogEntry]
    metrics: SystemMetrics
    alerts: List[Alert]
    step_count: int


class Action(BaseModel):
    action_type: str
    target: Optional[str] = None


class Reward(BaseModel):
    value: float


# ---------------- EPISODE STATE ----------------

class EpisodeState(BaseModel):
    visible_logs: List[LogEntry]
    all_logs: List[LogEntry]
    metrics: SystemMetrics
    alerts: List[Alert]

    step_count: int
    max_steps: int

    services_inspected: List[str]
    keywords_filtered: List[str]

    root_cause_marked: Optional[str]
    classification_marked: Optional[str]
    resolution_action: Optional[str]

    wrong_action_count: int
    destructive_action_count: int

    actions_history: List