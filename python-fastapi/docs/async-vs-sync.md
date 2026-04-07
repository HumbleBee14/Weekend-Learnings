# async def vs def in FastAPI

Both work in FastAPI, but they behave very differently under the hood.

## What is async/await?

async/await is a way to write **non-blocking code that looks like normal sequential code**.

### The problem it solves

```python
# Without async — imagine a restaurant with ONE waiter
def handle_request():
    data = db.query("SELECT ...")   # waiter stands at the kitchen doing NOTHING for 200ms
    return data                      # only then goes to serve the next customer

# The waiter (thread) is BLOCKED. Can't serve anyone else while waiting for the kitchen.
```

```python
# With async — the waiter takes the order, walks away, comes back when food is ready
async def handle_request():
    data = await db.query("SELECT ...")  # waiter drops off the order, goes to serve others
    return data                           # comes back when kitchen signals "food ready"
```

`await` is the keyword that means **"I'm going to wait for this, but let others do work while I wait."**

### The Java/Node mapping

```
Python:   async def + await
Java:     CompletableFuture.thenApply()  or  Spring WebFlux Mono/Flux
Node.js:  async function + await           (identical syntax)
```

```python
# Python
async def get_user(user_id):
    user = await db.fetch_one("SELECT * FROM users WHERE id = $1", user_id)
    return user
```

```java
// Java CompletableFuture (ugly, callback-style)
CompletableFuture<User> getUser(int userId) {
    return db.fetchOne("SELECT * FROM users WHERE id = ?", userId)
             .thenApply(row -> mapToUser(row));
}

// Java with virtual threads (Project Loom) — looks sync but isn't blocking OS threads
User getUser(int userId) {
    return db.fetchOne("SELECT * FROM users WHERE id = ?", userId);  // virtual thread parks here
}
```

```javascript
// Node.js — identical to Python
async function getUser(userId) {
    const user = await db.fetchOne("SELECT * FROM users WHERE id = $1", userId);
    return user;
}
```

Python's async/await is basically Node.js syntax with the same single-threaded event loop model.

### Rules of async/await

1. `await` can **only** be used inside an `async def` function
2. You can only `await` things that are "awaitable" (coroutines, Futures, Tasks)
3. `async def` functions return a coroutine — they don't run until awaited

```python
# This does NOT run the function:
result = get_user(42)          # result is a coroutine object, not the actual user!

# This runs it:
result = await get_user(42)    # NOW it executes and returns the user
```

Java equivalent of this mistake: calling a method that returns `CompletableFuture` but never calling `.get()` or `.join()` — you have the Future object, not the result.

### Nested async/await

When you have multiple async calls, you chain them naturally:

```python
# Sequential — each waits for the previous one
async def get_user_with_orders(user_id: int):
    user = await db.fetch_one("SELECT * FROM users WHERE id = $1", user_id)
    orders = await db.fetch_all("SELECT * FROM orders WHERE user_id = $1", user_id)
    latest_payment = await payment_service.get_latest(user_id)
    return {"user": user, "orders": orders, "payment": latest_payment}
# Total time: user_query + orders_query + payment_call (added up, one after another)
```

But `orders` and `latest_payment` don't depend on each other — why wait sequentially?

```python
import asyncio

# Parallel — run independent tasks concurrently
async def get_user_with_orders(user_id: int):
    user = await db.fetch_one("SELECT * FROM users WHERE id = $1", user_id)

    # These two don't depend on each other — run them at the same time
    orders, latest_payment = await asyncio.gather(
        db.fetch_all("SELECT * FROM orders WHERE user_id = $1", user_id),
        payment_service.get_latest(user_id),
    )
    return {"user": user, "orders": orders, "payment": latest_payment}
# Total time: user_query + max(orders_query, payment_call) — faster!
```

```java
// Java equivalent of asyncio.gather:
CompletableFuture.allOf(ordersFuture, paymentFuture).join();
```

```javascript
// Node.js equivalent of asyncio.gather:
const [orders, payment] = await Promise.all([getOrders(userId), getPayment(userId)]);
```

### Deeper nesting — async functions calling async functions

```python
async def get_enriched_order(order_id: int):
    order = await db.fetch_one("SELECT * FROM orders WHERE id = $1", order_id)
    product = await get_product_details(order["product_id"])    # calls another async fn
    shipping = await get_shipping_status(order["tracking_id"])  # and another
    return {**order, "product": product, "shipping": shipping}

async def get_product_details(product_id: int):
    product = await db.fetch_one("SELECT * FROM products WHERE id = $1", product_id)
    reviews = await review_service.get_reviews(product_id)  # async all the way down
    return {**product, "reviews": reviews}

async def get_shipping_status(tracking_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://shipping-api.com/track/{tracking_id}")
    return response.json()
```

Every `await` is a point where the event loop can go do other work. The call stack can be as deep as you want — as long as every layer is `async def` and uses `await`, it stays non-blocking.

**Think of it as:** `await` = "I'm pausing MY execution here, but the event loop is free to handle other requests while I wait."

## What FastAPI does internally

```python
# async — runs directly on the main event loop (like Node.js)
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# sync — FastAPI runs it in a separate thread pool (like Tomcat)
@app.get("/items/{item_id}")
def get_items(item_id: int):
    return {"id": item_id}
```

```
async def endpoint():     →  runs on event loop (single thread, non-blocking)
                             like Express/Node.js handlers

def endpoint():           →  runs in a threadpool via run_in_executor()
                             like Spring MVC on Tomcat (thread-per-request)
```

### Where does code actually run — the full picture

```
Uvicorn starts
    │
    ▼
┌── Main Thread (event loop) ─────────────────────────┐
│                                                       │
│   async def health():        ← runs HERE directly     │
│       return {"ok": True}                             │
│                                                       │
│   async def get_user():      ← runs HERE directly     │
│       user = await db.fetch()  ← at this "await",     │
│       return user                this thread handles   │
│                                  other requests        │
│                                                       │
│   def get_items():           ← NOT here. Delegated ──┼──► Thread Pool
│       items = db.query()                              │    ├─ OS Thread 1: get_items()
│       return items                                    │    ├─ OS Thread 2: (another sync call)
│                                                       │    └─ OS Thread 3: (another sync call)
└───────────────────────────────────────────────────────┘
```

| | Where does the code run? | Thread model |
|---|---|---|
| `async def` | Main event loop thread | Single thread, shared via `await` points |
| `def` | Thread pool (OS threads) | Separate thread per request from pool |

`async def` does NOT get its own thread. One thread juggles thousands of requests by switching at every `await`. `def` gets a real OS thread from the pool per request.

### What does "runs in a threadpool via run_in_executor" actually mean?

Uvicorn's main thread runs an **event loop** (single thread, like Node.js). If you write `async def`, your code runs directly on that one thread. Great for non-blocking I/O, terrible if you block it.

But what about plain `def` endpoints? FastAPI can't run blocking code on the event loop — it would freeze everything. So it does this behind the scenes:

```python
# What you write:
@app.get("/users")
def get_users():
    users = db.query("SELECT * FROM users")  # blocking call
    return users

# What FastAPI actually does internally:
@app.get("/users")
async def get_users_wrapper():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,          # None = use default ThreadPoolExecutor
        get_users      # your sync function
    )
    return result
```

`run_in_executor()` takes your blocking function and runs it on a **separate thread from a thread pool**, then `await`s the result. The event loop stays free.

```
                         ┌─────────────────────────────────┐
  Event Loop (1 thread)  │  handles async def endpoints     │
  ───────────────────────│  handles incoming connections     │
                         │  coordinates everything           │
                         └───────────┬─────────────────────┘
                                     │
                          run_in_executor()
                                     │
                         ┌───────────▼─────────────────────┐
  Thread Pool (N threads)│  Thread 1: def get_users()       │
                         │  Thread 2: def get_orders()      │
                         │  Thread 3: def get_products()    │
                         │  (each blocking call gets a thread)
                         └─────────────────────────────────┘
```

**This is exactly the Tomcat model.** Tomcat has a thread pool, each request gets a thread, the thread blocks on I/O. FastAPI does the same thing — but only for plain `def` endpoints. `async def` endpoints skip the thread pool entirely and run on the event loop.

### So why not just use `def` for everything?

Threads have overhead — context switching, memory per thread, GIL contention in Python. With `async def` + proper async libraries, one thread handles thousands of concurrent connections. With `def` + thread pool, you're limited by thread pool size (default ~40 threads).

```
async def + await    →  1 thread, 10,000 concurrent connections  (Node.js model)
def + threadpool     →  40 threads, 40 concurrent connections    (Tomcat model)
```

For high-concurrency APIs, `async def` wins. For simple CRUD with blocking DB drivers, `def` is perfectly fine and simpler.

## When to use which

| Your function does... | Use | Why |
|---|---|---|
| CPU-only work, returns data directly | `def` or `async def` | Both fine, no I/O to wait on |
| `await`-able I/O (async DB, async HTTP client) | `async def` | You need `await`, which only works in async functions |
| Blocking I/O (sync DB driver, file read, `requests.get()`) | **`def`** (no async!) | If you use `async def` with blocking calls, you freeze the entire event loop |

## The dangerous mistake

```python
import requests  # sync HTTP library (like Java's RestTemplate)

# WRONG — blocks the entire event loop, all other requests wait
@app.get("/external")
async def get_external():
    response = requests.get("https://api.example.com/data")  # blocks!
    return response.json()

# RIGHT — FastAPI runs this in a thread, event loop stays free
@app.get("/external")
def get_external():
    response = requests.get("https://api.example.com/data")  # blocks, but in its own thread
    return response.json()

# ALSO RIGHT — use an async HTTP library
@app.get("/external")
async def get_external():
    async with httpx.AsyncClient() as client:  # async library (like WebClient in WebFlux)
        response = await client.get("https://api.example.com/data")
    return response.json()
```

**Java analogy:** using `async def` with blocking I/O is like doing `RestTemplate.getForObject()` inside a Spring WebFlux handler on the Netty event loop — you'd freeze everything.
