from __future__ import annotations

import logging
import os
from datetime import datetime

import httpx

log = logging.getLogger(__name__)

WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")


def _send(event: str, data: dict) -> bool:
    if not WEBHOOK_URL:
        return False
    payload = {"event": event, "data": data, "timestamp": datetime.now().isoformat()}
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(WEBHOOK_URL, json=payload)
            return resp.status_code < 400
    except Exception as e:
        log.debug("Webhook failed for %s: %s", event, e)
        return False


def notify_facts_expiring(facts: list[dict]) -> bool:
    if not facts:
        return False
    return _send("facts_expiring", {"count": len(facts), "facts": facts[:10]})


def notify_task_due(task: dict) -> bool:
    return _send("task_due", task)


def notify_ingestion_complete(result: dict) -> bool:
    return _send("ingestion_complete", result)


def notify_maintenance(stats: dict) -> bool:
    return _send("maintenance_complete", stats)
