"""
High-Performance LLM Serving with Advanced Optimizations

This module implements a production-ready LLM serving system with multiple optimization
techniques including:
- Request batching and prioritization queue
- Two-level caching (exact Redis + semantic FAISS)
- Streaming token generation
- Request timeouts and error handling
- FastAPI endpoints with proper resource lifecycle management

The system is designed for high-throughput, low-latency LLM inference with efficient
resource utilization through batching, caching, and asynchronous processing.

Suitable for production deployments requiring optimal performance and cost efficiency.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import faiss
import orjson
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from redis.asyncio import Redis
from sentence_transformers import SentenceTransformer
from vllm import AsyncLLMEngine, EngineArgs, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Query(BaseModel):
    text: str
    params: Optional[dict] = None

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson.dumps


@dataclass
class QueueItem:
    query: Query
    future: asyncio.Future
    added_time: float
    priority: int = 0


class RequestQueue:
    def __init__(
        self,
        cache_manager: "CacheManager",
        token_manager: "TokenManager",
        max_batch_size: int = 32,
        max_wait_time: float = 0.1,
        max_queue_size: int = 1000,
    ):
        self.cache_manager = cache_manager
        self.token_manager = token_manager
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.is_running = False
        self.processing_task: Optional[asyncio.Task] = None

    async def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_queue())
        logger.info("Request queue processor started")

    async def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        if self.processing_task:
            try:
                await self.processing_task
            except Exception as e:
                logger.error(f"Error during queue shutdown: {e}")
        logger.info("Request queue processor stopped")

    async def add_request(self, query: Query, priority: int = 0) -> asyncio.Future:
        if not self.is_running:
            raise RuntimeError("Queue processor is not running")

        future = asyncio.Future()
        try:
            await self.queue.put(QueueItem(query=query, future=future, added_time=time.time(), priority=priority))
            return future
        except asyncio.QueueFull:
            raise RuntimeError("Request queue is full")

    async def _process_queue(self):
        while self.is_running:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch)
                else:
                    await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in queue processing: {e}")
                await asyncio.sleep(1)  # Prevent rapid retries on persistent errors

    async def _collect_batch(self) -> list[QueueItem]:
        batch = []
        try:
            # Get first item
            first_item = await asyncio.wait_for(self.queue.get(), timeout=self.max_wait_time)
            batch.append(first_item)

            # Collect additional items
            batch_start_time = time.time()
            while len(batch) < self.max_batch_size:
                if time.time() - batch_start_time > self.max_wait_time:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=0.01)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

        except asyncio.TimeoutError:
            return []

        return sorted(batch, key=lambda x: (-x.priority, x.added_time))

    async def _process_batch(self, batch: list[QueueItem]):
        try:
            # Check cache first
            queries = [item.query for item in batch]
            cached_responses = await self.cache_manager.get_exact_match(queries)

            # Process items
            for item, cached_response in zip(batch, cached_responses):
                try:
                    if cached_response:
                        if not item.future.done():
                            item.future.set_result(cached_response)
                        continue

                    # Check semantic cache
                    semantic_match = await self.cache_manager.get_semantic_match(item.query)
                    if semantic_match:
                        if not item.future.done():
                            item.future.set_result(semantic_match)
                        continue

                    # Generate new response
                    response = ""
                    async for chunk in self.token_manager.generate_stream(item.query):
                        response += chunk
                    await self.cache_manager.cache_response(item.query, response)
                    if not item.future.done():
                        item.future.set_result(response)

                except Exception as e:
                    logger.error(f"Error processing queue item: {e}")
                    if not item.future.done():
                        item.future.set_exception(e)

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)


class SemanticCache:
    def __init__(self, dimension: int = 384, max_cache_size: int = 10000):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(dimension)
        self.cached_responses = {}
        self.dimension = dimension
        self.max_cache_size = max_cache_size

    def add_to_cache(self, text: str, response: str):
        if self.index.ntotal >= self.max_cache_size:
            # Simple FIFO cache eviction
            self.index = faiss.IndexFlatL2(self.dimension)
            self.cached_responses.clear()

        embedding = self.encoder.encode([text])[0].astype("float32")
        self.index.add(embedding.reshape(1, -1))
        self.cached_responses[self.index.ntotal - 1] = response

    async def find_similar(self, text: str, threshold: float = 0.9) -> Optional[str]:
        if self.index.ntotal == 0:
            return None

        embedding = self.encoder.encode([text])[0].astype("float32")
        D, I = self.index.search(embedding.reshape(1, -1), 1)

        if D[0][0] < threshold:
            return self.cached_responses[I[0][0]]
        return None


class CacheManager:
    def __init__(self, redis_url: str = "redis://localhost:6800"):
        self.redis = Redis.from_url(redis_url, encoding="utf-8", decode_responses=True, max_connections=20)
        self.semantic_cache = SemanticCache()
        self.cache_ttl = 3600  # 1 hour

    async def get_exact_match(self, queries: list[Query]) -> list[Optional[str]]:
        pipe = self.redis.pipeline()
        for query in queries:
            key = f"cache:{query.text}"
            pipe.get(key)
        return await pipe.execute()

    async def get_semantic_match(self, query: Query) -> Optional[str]:
        return await self.semantic_cache.find_similar(query.text)

    async def cache_response(self, query: Query, response: str):
        key = f"cache:{query.text}"
        pipe = self.redis.pipeline()
        pipe.set(key, response, ex=self.cache_ttl)
        await pipe.execute()

        self.semantic_cache.add_to_cache(query.text, response)

    async def get_cache_stats(self):
        return {"exact_cache_size": await self.redis.dbsize(), "semantic_cache_size": self.semantic_cache.index.ntotal}

    async def clear_cache(self):
        await self.redis.flushall()
        self.semantic_cache = SemanticCache()


class TokenManager:
    def __init__(self, model_name: str):
        self.engine = AsyncLLMEngine.from_engine_args(
            engine_args=EngineArgs(
                model=model_name,
                tensor_parallel_size=1,
                max_num_batched_tokens=4096,
            )
        )

    async def generate_stream(self, query: Query) -> AsyncGenerator[str, None]:
        params = (
            SamplingParams(**query.params)
            if query.params
            else SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)
        )

        try:
            async for output in self.engine.generate(query.text, params, 0):
                if output.finished:
                    yield output.outputs[0].text
                else:
                    yield output.outputs[0].text + "<|CONTINUE|>"
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize components
    app.state.cache = CacheManager()
    app.state.token_mgr = TokenManager("mistralai/Mixtral-8x7B-Instruct-v0.1")
    app.state.request_queue = RequestQueue(cache_manager=app.state.cache, token_manager=app.state.token_mgr)

    # Start request queue
    await app.state.request_queue.start()

    yield

    # Cleanup
    await app.state.request_queue.stop()
    await app.state.cache.redis.close()


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate_response(query: Query):
    try:
        # Add request to queue and wait for result
        future = await app.state.request_queue.add_request(query)
        try:
            response = await asyncio.wait_for(future, timeout=30.0)  # Add timeout
            return StreamingResponse(iter([response]), media_type="text/plain")
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Request timeout")

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_batch")
async def generate_batch(queries: list[Query]):
    try:
        # Add all queries to queue
        futures = []
        for query in queries:
            try:
                future = await app.state.request_queue.add_request(query)
                futures.append(future)
            except Exception as e:
                logger.error(f"Error adding query to queue: {e}")
                raise

        # Wait for all results with timeout
        try:
            responses = await asyncio.wait_for(
                asyncio.gather(*futures),
                timeout=60.0,  # Adjust timeout based on your needs
            )

            return [{"response": response, "source": "queue"} for response in responses]

        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Batch processing timeout")

    except Exception as e:
        logger.error(f"Error processing batch request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats")
async def get_cache_stats():
    return await app.state.cache.get_cache_stats()


@app.post("/cache/clear")
async def clear_cache():
    await app.state.cache.clear_cache()
    return {"status": "Cache cleared"}
