import logging
import requests
from typing import Protocol, Optional, Dict, Any
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# --- Validated Payloads ---

class JobEventPayload(BaseModel):
    job_id: str
    event_type: str
    details: Optional[Dict[str, Any]] = None

class JobProgressPayload(BaseModel):
    job_id: str
    frames_inspected: int
    valid_detections: int
    invalid_detections: int

# --- Reporter Interface ---

class Reporter(Protocol):
    def report_start(self, job_id: str) -> None: ...
    def report_progress(self, job_id: str, inspected: int, valid: int, invalid: int) -> None: ...
    def report_success(self, job_id: str, results_summary: Dict[str, Any]) -> None: ...
    def report_failure(self, job_id: str, error_msg: str) -> None: ...

# --- Implementations ---

class DummyReporter:
    """A no-op reporter used when reporting is disabled or during tests."""
    def report_start(self, job_id: str) -> None: pass
    def report_progress(self, job_id: str, inspected: int, valid: int, invalid: int) -> None: pass
    def report_success(self, job_id: str, results_summary: Dict[str, Any]) -> None: pass
    def report_failure(self, job_id: str, error_msg: str) -> None: pass

class HttpMockApiReporter:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.events_url = f"{self.base_url}/api/v1/jobs/events"
        self.progress_url = f"{self.base_url}/api/v1/jobs/progress"
        self.timeout = 5.0 # Seconds

    def _post_json(self, url: str, payload_dict: Dict[str, Any]) -> None:
        try:
            resp = requests.post(url, json=payload_dict, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            # We explicitly catch reporting failures and log them, 
            # so they do NOT crash the video pipeline.
            logger.warning(f"Reporting failure to {url}: {e}")

    def report_start(self, job_id: str) -> None:
        payload = JobEventPayload(job_id=job_id, event_type="STARTED")
        self._post_json(self.events_url, payload.model_dump())

    def report_progress(self, job_id: str, inspected: int, valid: int, invalid: int) -> None:
        payload = JobProgressPayload(
            job_id=job_id,
            frames_inspected=inspected,
            valid_detections=valid,
            invalid_detections=invalid
        )
        self._post_json(self.progress_url, payload.model_dump())

    def report_success(self, job_id: str, results_summary: Dict[str, Any]) -> None:
        payload = JobEventPayload(
            job_id=job_id, 
            event_type="SUCCESS",
            details=results_summary
        )
        self._post_json(self.events_url, payload.model_dump())

    def report_failure(self, job_id: str, error_msg: str) -> None:
        payload = JobEventPayload(
            job_id=job_id, 
            event_type="FAILED",
            details={"error": error_msg}
        )
        self._post_json(self.events_url, payload.model_dump())
