"""
Replace ARC puzzle-id indices with stable buckets derived from the query string.

TRM uses sparse puzzle embeddings; for QA we treat each query as its own "puzzle id"
via hashing into [0, num_identifiers).
"""

from __future__ import annotations

import zlib


def query_to_puzzle_id(query: str, num_puzzle_identifiers: int, blank_id: int = 0) -> int:
    """
    Map query text to an integer in [0, num_identifiers).

    Uses zlib CRC32 for speed and stability (not cryptographic).
    """
    if num_puzzle_identifiers <= 0:
        return blank_id
    h = zlib.crc32(query.encode("utf-8", errors="replace")) & 0xFFFFFFFF
    return int(h % num_puzzle_identifiers)


def batch_query_puzzle_ids(queries: list[str], num_puzzle_identifiers: int) -> list[int]:
    return [query_to_puzzle_id(q, num_puzzle_identifiers) for q in queries]
