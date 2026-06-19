"""Confidence queue: routing, resolution, consensus detection."""

from __future__ import annotations

import json

from drbrain.storage.database import Database


def route_item(
    db: Database,
    source_paper: str,
    item_type: str,
    item_data: dict,
    confidence: float,
    weak_threshold: float = 0.7,
    auto_accept: float = 0.9,
) -> dict:
    """Route an extracted item based on its confidence."""
    if confidence >= auto_accept:
        return {"action": "accepted", "queue_id": None}
    elif confidence >= weak_threshold:
        return {"action": "weak", "queue_id": None}
    else:
        qid = db.insert_queue_item(
            source_paper,
            item_type,
            json.dumps(item_data),
            confidence,
        )
        return {"action": "queued", "queue_id": qid}


def check_consensus(
    db: Database, label: str, min_papers: int = 3, min_confidence: float = 0.8
) -> bool:
    """Check if a concept label has consensus (N+ papers with high confidence)."""
    row = db.conn.execute(
        "SELECT COUNT(DISTINCT c.local_id), AVG(c.confidence) FROM concepts c WHERE c.label = ?",
        (label,),
    ).fetchone()
    if row is None:
        return False
    count, avg_conf = row
    return count >= min_papers and avg_conf >= min_confidence


def resolve_accept(db: Database, queue_id: int) -> None:
    """Accept a queue item. Auto-accept matching items if concept has consensus."""
    item = db.conn.execute(
        "SELECT item_data FROM confidence_queue WHERE queue_id = ?", (queue_id,)
    ).fetchone()
    if item is None:
        return

    data = json.loads(item[0])
    label = data.get("label", "")

    db.accept_queue_item(queue_id)

    if label and check_consensus(db, label):
        db.accept_queue_by_label(label)

    db.commit()


def resolve_reject(db: Database, queue_id: int) -> None:
    """Reject a queue item."""
    db.reject_queue_item(queue_id)
    db.commit()


def resolve_all(
    db: Database,
    action: str,
    type_filter: str | None = None,
    max_conf: float | None = None,
) -> dict:
    """Batch resolve all pending queue items matching filters."""
    sql = "SELECT queue_id, item_type, confidence FROM confidence_queue WHERE status = 'pending'"
    params: list = []
    if type_filter:
        sql += " AND item_type = ?"
        params.append(type_filter)
    if max_conf is not None:
        sql += " AND confidence <= ?"
        params.append(max_conf)

    rows = db.conn.execute(sql, params).fetchall()
    count = 0
    for qid, _item_type, _conf in rows:
        if action == "accept":
            db.accept_queue_item(qid)
        else:
            db.reject_queue_item(qid)
        count += 1

    if count > 0:
        db.commit()

    return {"count": count}
