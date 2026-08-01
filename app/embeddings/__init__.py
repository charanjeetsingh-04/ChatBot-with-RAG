"""Embedding model interface and implementations.

Used by both ingestion (encode chunks) and retrieval (encode query). Kept in its
own package so swapping providers is a one-file change.
"""
