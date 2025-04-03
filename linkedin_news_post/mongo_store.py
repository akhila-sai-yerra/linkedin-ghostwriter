from __future__ import annotations

import asyncio
import logging
import pymongo
import uuid
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Optional, Dict, Any, Tuple, List, Union, Iterable

# Imports from your store contract
from langgraph.store.base import (
    BaseStore,
    Item,
    SearchItem,
    GetOp,
    SearchOp,
    PutOp,
    ListNamespacesOp,
    InvalidNamespaceError,
)


# Custom get_text_at_path that supports dot-notation
def get_text_at_path(doc: dict, fields: List[str]) -> str:
    """Extract text from a document following each field's dot-notation.
    For example, if fields is ['content.article', 'summary'] and doc is
    {'content': {'article': 'Hello world'}, 'summary': 'Greetings'},
    then it returns 'Hello world Greetings'.
    """
    results = []
    for field in fields:
        parts = field.split(".")
        current = doc
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                current = None
                break
        if isinstance(current, str) and current.strip():
            results.append(current.strip())
    return " ".join(results)


# Sentinel for a value that is "not provided"
class _NotProvidedSentinel:
    pass


NOT_PROVIDED: _NotProvidedSentinel = _NotProvidedSentinel()


class MongoDBBaseStore(BaseStore):
    """
    MongoDB-based persistent store that supports both standard (text)
    search and semantic (vector) search.

    When an index configuration is provided (including an 'embed'
    embedding function), the store computes and stores embeddings during put() operations,
    then uses a knnBeta aggregation stage for search().

    IMPORTANT: Your Atlas Search index must be configured so that the "embedding" field is mapped as a knnVector.
    For example, in Atlas use a mapping similar to:

      {
        "mappings": {
          "dynamic": false,
          "fields": {
            "embedding": {
              "type": "knnVector",
              "dimensions": 1536,
              "similarity": "cosine"
            }
          }
        }
      }

    Also ensure that the "index_name" in your index_config (e.g., "store_index") matches the Atlas index.
    
    Note: The search() operation is read-only; inserted documents are not updated by search,
    and Atlas Search indexes are updated asynchronously.
    """

    def __init__(
        self,
        mongo_url: str,
        db_name: str = "checkpointing_db",
        collection_name: str = "store",
        ttl_support: bool = False,
        index_config: Optional[Dict[str, Any]] = None,
    ):
        self._client = pymongo.MongoClient(mongo_url)
        self._db = self._client[db_name]
        self._collection = self._db[collection_name]
        self._ttl_support = ttl_support
        self.supports_ttl = ttl_support

        if ttl_support:
            self._collection.create_index("expiration", expireAfterSeconds=0)

        # Create a text index on "value" for fallback queries
        self._collection.create_index([("value", "text")], name="value_text_index")

        self._index_config = index_config
        if index_config and "embed" in index_config:
            self.semantic_enabled = True
            self._embedding_fn = index_config["embed"]
            self.index_name = index_config.get("index_name", "langchain_vsearch_index")
        else:
            self.semantic_enabled = False
            self._embedding_fn = None
            self.index_name = None

    def _namespace_query(self, namespace: Tuple[str, ...]) -> Dict[str, Any]:
        return {"namespace": list(namespace)}

    def _namespace_prefix_query(self, namespace_prefix: Tuple[str, ...]) -> Dict[str, Any]:
        query: Dict[str, Any] = {}
        for i, part in enumerate(namespace_prefix):
            query[f"namespace.{i}"] = part
        return query

    def _compute_expiration(self, ttl: Optional[float]) -> Optional[datetime]:
        if ttl is None:
            return None
        return datetime.now(timezone.utc) + timedelta(minutes=ttl)

    # Synchronous Methods
    def get(
        self,
        namespace: Tuple[str, ...],
        key: str,
        *,
        refresh_ttl: Optional[bool] = None,
    ) -> Optional[Item]:
        q = {**self._namespace_query(namespace), "key": key}
        doc = self._collection.find_one(q)
        if doc is None:
            return None
        if refresh_ttl and self._ttl_support and doc.get("expiration"):
            new_exp = self._compute_expiration(10)
            self._collection.update_one(q, {"$set": {"expiration": new_exp}})
            doc["expiration"] = new_exp
        return Item(
            value=doc["value"],
            key=doc.get("logical_key", doc["key"]),
            namespace=tuple(doc["namespace"]),
            created_at=doc.get("created"),
            updated_at=doc.get("created"),
        )

    def search(
        self,
        namespace_prefix: Tuple[str, ...],
        *,
        query: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: Optional[bool] = None,
    ) -> List[SearchItem]:
        results: List[SearchItem] = []
        if self.semantic_enabled:
            # Always use semantic (vector) search when enabled
            query_vector = self._embedding_fn(query or "")
            search_stage = {
                "$search": {
                    "index": self.index_name,
                    "knnBeta": {
                        "vector": query_vector,
                        "path": "embedding",
                        "k": limit,
                    },
                }
            }
            pipeline: List[Dict[str, Any]] = [search_stage]
            ns_filter = self._namespace_prefix_query(namespace_prefix)
            if filter:
                for k, v in filter.items():
                    ns_filter[f"value.{k}"] = v
            if ns_filter:
                pipeline.append({"$match": ns_filter})
            pipeline.append({"$sort": {"score": -1}})
            pipeline.extend(
                [
                    {"$skip": offset},
                    {"$limit": limit},
                    {
                        "$project": {
                            "namespace": 1,
                            "key": 1,
                            "value": 1,
                            "created": 1,
                            "score": {"$meta": "searchScore"},
                        }
                    },
                ]
            )
            try:
                cursor = self._collection.aggregate(pipeline)
                docs = list(cursor)
            except pymongo.errors.OperationFailure as e:
                logging.error("Error running semantic search: %s", e)
                docs = []
            seen_keys = set()
            for doc in docs:
                if refresh_ttl and self._ttl_support and doc.get("expiration"):
                    new_exp = self._compute_expiration(10)
                    self._collection.update_one(
                        {"_id": doc["_id"]}, {"$set": {"expiration": new_exp}}
                    )
                    doc["expiration"] = new_exp
                item = SearchItem(
                    namespace=tuple(doc["namespace"]),
                    key=doc.get("logical_key", doc["key"]),
                    value=doc["value"],
                    created_at=doc.get("created"),
                    updated_at=doc.get("created"),
                    score=doc.get("score"),
                )
                results.append(item)
                seen_keys.add(doc.get("logical_key", doc["key"]))
            # Optionally, if fewer documents returned than limit, fill with fallback text search
            if len(results) < limit:
                remaining = limit - len(results)
                fallback_q = self._namespace_prefix_query(namespace_prefix)
                if filter:
                    for k, v in filter.items():
                        fallback_q[f"value.{k}"] = v
                if query:
                    fallback_q["$text"] = {"$search": query}
                projection: Dict[str, Any] = {
                    "namespace": 1,
                    "key": 1,
                    "value": 1,
                    "created": 1,
                }
                if query:
                    projection["score"] = {"$meta": "textScore"}
                fallback_cursor = (
                    self._collection.find(fallback_q, projection=projection)
                    .sort([("score", {"$meta": "textScore"})])
                    .skip(offset)
                    .limit(remaining)
                )
                for doc in fallback_cursor:
                    key_val = doc.get("logical_key", doc["key"])
                    if key_val in seen_keys:
                        continue
                    item = SearchItem(
                        namespace=tuple(doc["namespace"]),
                        key=doc.get("logical_key", doc["key"]),
                        value=doc["value"],
                        created_at=doc.get("created"),
                        updated_at=doc.get("created"),
                        score=doc.get("score"),
                    )
                    results.append(item)
                    seen_keys.add(key_val)
                    if len(results) >= limit:
                        break
            return results
        else:
            # Fallback: basic text search when semantic search is disabled
            q = self._namespace_prefix_query(namespace_prefix)
            if filter:
                for k, v in filter.items():
                    q[f"value.{k}"] = v
            if query:
                q["$text"] = {"$search": query}
            projection: Dict[str, Any] = {"namespace": 1, "key": 1, "value": 1, "created": 1}
            if query:
                projection["score"] = {"$meta": "textScore"}
            cursor = (
                self._collection.find(q, projection=projection)
                .sort([("score", {"$meta": "textScore"})])
                .skip(offset)
                .limit(limit)
            )
            for doc in cursor:
                results.append(
                    SearchItem(
                        namespace=tuple(doc["namespace"]),
                        key=doc.get("logical_key", doc["key"]),
                        value=doc["value"],
                        created_at=doc.get("created"),
                        updated_at=doc.get("created"),
                        score=doc.get("score"),
                    )
                )
            return results

    def put(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: Dict[str, Any],
        index: Optional[Union[bool, List[str]]] = None,
        *,
        ttl: Union[Optional[float], _NotProvidedSentinel] = NOT_PROVIDED,
    ) -> None:
        unique_key = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        doc: Dict[str, Any] = {
            "namespace": list(namespace),
            "key": unique_key,
            "logical_key": key,
            "value": value,
            "created": now,
        }
        if ttl is not NOT_PROVIDED and self._ttl_support:
            doc["expiration"] = self._compute_expiration(ttl) if ttl is not None else None
        if index is not None:
            doc["indexed"] = index

        # When semantic search is enabled, extract text and compute embedding
        if index is not False and self.semantic_enabled:
            # Get the fields from config; default to ["$"] if not provided
            fields: List[str] = (
                self._index_config.get("fields", ["$"]) if self._index_config else ["$"]
            )
            text = get_text_at_path(value, fields)
            logging.info("Extracted text for embedding using fields %s: %s", fields, text)
            if text.strip():
                embedding_vector = self._embedding_fn(text)
                logging.info("Created embedding vector of length %d", len(embedding_vector))
                doc["embedding"] = embedding_vector
            else:
                logging.warning(
                    "No text extracted for embedding; document will not have an embedding."
                )

        self._collection.insert_one(doc)
        logging.info("Inserted document with key %s into collection", unique_key)

    def delete(self, namespace: Tuple[str, ...], key: str) -> None:
        q = {**self._namespace_query(namespace), "key": key}
        self._collection.delete_one(q)

    def list_namespaces(
        self,
        *,
        prefix: Optional[Tuple[str, ...]] = None,
        suffix: Optional[Tuple[str, ...]] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Tuple[str, ...]]:
        q: Dict[str, Any] = {}
        if prefix:
            q.update(self._namespace_prefix_query(prefix))
        namespaces = self._collection.distinct("namespace", q)
        if suffix:
            def matches_suffix(ns: List[str]) -> bool:
                return (
                    tuple(ns[-len(suffix) :]) == suffix if len(ns) >= len(suffix) else False
                )

            namespaces = [ns for ns in namespaces if matches_suffix(ns)]
        if max_depth is not None:
            namespaces = [ns[:max_depth] for ns in namespaces]
        unique_namespaces = sorted({tuple(ns) for ns in namespaces})
        return unique_namespaces[offset : offset + limit]

    def batch(self, ops: Iterable[Any]) -> List[Any]:
        results = []
        for op in ops:
            if isinstance(op, GetOp):
                res = self.get(op.namespace, op.key, refresh_ttl=op.refresh_ttl)
            elif isinstance(op, SearchOp):
                res = self.search(
                    op.namespace_prefix,
                    query=op.query,
                    filter=op.filter,
                    limit=op.limit,
                    offset=op.offset,
                    refresh_ttl=op.refresh_ttl,
                )
            elif isinstance(op, PutOp):
                if op.value is not None:
                    res = self.put(op.namespace, op.key, op.value, index=op.index, ttl=op.ttl)
                else:
                    res = self.delete(op.namespace, op.key)
            elif isinstance(op, ListNamespacesOp):
                prefix = None
                suffix = None
                if op.match_conditions:
                    for cond in op.match_conditions:
                        if cond.match_type == "prefix":
                            prefix = cond.path
                        elif cond.match_type == "suffix":
                            suffix = cond.path
                res = self.list_namespaces(
                    prefix=prefix,
                    suffix=suffix,
                    max_depth=op.max_depth,
                    limit=op.limit,
                    offset=op.offset,
                )
            else:
                res = None
            results.append(res)
        return results

    async def abatch(self, ops: Iterable[Any]) -> List[Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.batch, ops))

    async def aget(
        self,
        namespace: Tuple[str, ...],
        key: str,
        *,
        refresh_ttl: Optional[bool] = None,
    ) -> Optional[Item]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(self.get, namespace, key, refresh_ttl=refresh_ttl)
        )

    async def asearch(
        self,
        namespace_prefix: Tuple[str, ...],
        *,
        query: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: Optional[bool] = None,
    ) -> List[SearchItem]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(
                self.search,
                namespace_prefix,
                query=query,
                filter=filter,
                limit=limit,
                offset=offset,
                refresh_ttl=refresh_ttl,
            ),
        )

    async def aput(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: Dict[str, Any],
        index: Optional[Union[bool, List[str]]] = None,
        *,
        ttl: Union[Optional[float], _NotProvidedSentinel] = NOT_PROVIDED,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, partial(self.put, namespace, key, value, index, ttl=ttl)
        )

    async def adelete(self, namespace: Tuple[str, ...], key: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, partial(self.delete, namespace, key))

    async def alist_namespaces(
        self,
        *,
        prefix: Optional[Tuple[str, ...]] = None,
        suffix: Optional[Tuple[str, ...]] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Tuple[str, ...]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(
                self.list_namespaces,
                prefix=prefix,
                suffix=suffix,
                max_depth=max_depth,
                limit=limit,
                offset=offset,
            ),
        )