import os
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "olist_reviews")

if not QDRANT_URL:
    raise RuntimeError("QDRANT_URL is not set")

client_kwargs = {"url": QDRANT_URL, "timeout": 30}
if QDRANT_API_KEY and QDRANT_URL.lower().startswith("https"):
    client_kwargs["api_key"] = QDRANT_API_KEY

qc = QdrantClient(**client_kwargs)

print(f"Backfilling payload in collection '{COLLECTION}' at {QDRANT_URL}")

updated = 0
checked = 0
next_offset = None

while True:
    points, next_offset = qc.scroll(
        collection_name=COLLECTION,
        with_payload=True,
        with_vectors=False,
        limit=1000,
        offset=next_offset,
    )
    if not points:
        break
    upserts = []
    for p in points:
        checked += 1
        payload = (getattr(p, "payload", {}) or {}).copy()
        # Normalize keys possibly missing: rating, product_id
        rating = payload.get("rating")
        product_id = payload.get("product_id")
        # Sometimes metadata keys may be nested or named differently; try common fallbacks
        if rating is None:
            rating = payload.get("review_score")
            if rating is not None:
                payload["rating"] = rating
        if product_id is None:
            pid = payload.get("product_id") or payload.get("prod_id")
            if pid is not None:
                payload["product_id"] = pid
        if payload != getattr(p, "payload", {}):
            upserts.append(models.PointStruct(id=p.id, payload=payload))
    if upserts:
        qc.upsert(collection_name=COLLECTION, points=upserts)
        updated += len(upserts)
    if not next_offset:
        break

print(f"Checked {checked} points; updated {updated} payloads.")
