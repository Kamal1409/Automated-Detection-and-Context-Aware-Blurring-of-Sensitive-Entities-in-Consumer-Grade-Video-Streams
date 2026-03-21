import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


class JobManager:
    def __init__(self, max_workers=2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs = {}
        self._lock = threading.Lock()

    def create(self, job_id, message="Queued"):
        now = datetime.now().isoformat()
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "phase": "queued",
                "progress": 0.0,
                "message": message,
                "error": None,
                "result": None,
                "created_at": now,
                "updated_at": now,
                "finished_at": None,
            }

    def update(self, job_id, **patch):
        with self._lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id].update(patch)
            self._jobs[job_id]["updated_at"] = datetime.now().isoformat()

    def complete(self, job_id, result):
        self.update(
            job_id,
            status="completed",
            phase="completed",
            progress=100.0,
            message="Processing complete",
            result=result,
            finished_at=datetime.now().isoformat(),
        )

    def fail(self, job_id, message):
        self.update(
            job_id,
            status="failed",
            phase="failed",
            message=message,
            error=message,
            finished_at=datetime.now().isoformat(),
        )

    def get(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return dict(job)

    def submit(self, fn, *args, **kwargs):
        self._executor.submit(fn, *args, **kwargs)
