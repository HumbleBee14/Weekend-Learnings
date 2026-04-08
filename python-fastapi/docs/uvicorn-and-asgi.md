# Uvicorn & ASGI

## What is Uvicorn?

Uvicorn is the **server** that runs your FastAPI app. FastAPI itself is just the framework — it doesn't know how to listen on a port or handle HTTP connections. Uvicorn does that.

### Mapping to what you know

| Python (FastAPI) | Java (Spring Boot) | Node.js (Express) |
|------------------|--------------------|--------------------|
| Uvicorn | Embedded Tomcat | Node's built-in `http` module |
| FastAPI | Spring MVC / Spring WebFlux | Express.js |
| `uvicorn main:app --reload` | `mvn spring-boot:run` | `node app.js` / `nodemon app.js` |

In Spring Boot, Tomcat is embedded — you don't think about it. In Express, Node's HTTP server is baked in. In Python, the server is **separate** from the framework. You pick your server (Uvicorn, Hypercorn, Daphne) and point it at your app.

```bash
uvicorn main:app --reload
#       ^^^^ ^^^ ^^^^^^^^
#       |    |   auto-restart on file changes (like nodemon / spring-devtools)
#       |    the FastAPI() instance variable name in your code
#       the Python file (main.py)
```

## What is ASGI?

ASGI = **Asynchronous Server Gateway Interface**. It's a spec — a contract between the server (Uvicorn) and the framework (FastAPI).

Think of it like this:

| Ecosystem | The "contract" between server and framework |
|-----------|---------------------------------------------|
| Python (old) | WSGI — synchronous only (used by Flask, Django) |
| Python (new) | **ASGI** — supports async + sync (used by FastAPI, Django 3+) |
| Java | Servlet API (Tomcat ↔ Spring MVC), Reactive Streams (Netty ↔ WebFlux) |
| Node.js | No formal spec needed — everything is async by default |

### ASGI vs WSGI

WSGI (Web Server Gateway Interface) was Python's original standard. It's **synchronous** — each request gets a thread, that thread blocks until the response is ready, then moves on. No WebSockets, no long-lived connections, no concurrency within a single thread.

ASGI (Asynchronous Server Gateway Interface) is its successor. It's **asynchronous** — a single thread can handle thousands of concurrent connections by switching between them at `await` points. This enables:

- **Non-blocking I/O** — while waiting for a DB query or external API, the server handles other requests
- **WebSockets** — long-lived bidirectional connections (impossible with WSGI)
- **Server-Sent Events** — streaming responses
- **Better throughput** — fewer threads doing more work

```
WSGI (Flask, old Django):
  Request 1 ████████████░░░░░░░░ (thread 1 blocked on DB)
  Request 2 ░░░░░░░░░░░░████████ (thread 2 waits for thread 1 to free up)
  → sequential, one at a time per thread

ASGI (FastAPI, Django 3+):
  Request 1 ████░░░░████░░░░████ (event loop: work, await, work, await, work)
  Request 2 ░░░░████░░░░████░░░░ (fills in the gaps while Req 1 awaits)
  → concurrent, interleaved on one thread
```

| | WSGI | ASGI |
|---|---|---|
| Model | Thread-per-request (like Tomcat) | Event loop (like Node.js) |
| Concurrency | Limited by thread pool size | Thousands on a single thread |
| WebSockets | Not supported | Supported |
| Frameworks | Flask, Django (traditional) | FastAPI, Django 3+ (async views) |
| Java equivalent | Servlet API (blocking) | Reactive Streams / Netty (non-blocking) |
| Node equivalent | — | Native model (Node is async by default) |

```python
# WSGI world (Flask) — blocks the thread while waiting for DB
@app.route("/users")
def get_users():
    users = db.query("SELECT * FROM users")  # thread sits idle, waiting
    return users

# ASGI world (FastAPI) — thread is free while waiting for DB
@app.get("/users")
async def get_users():
    users = await db.query("SELECT * FROM users")  # thread does other work
    return users
```

### The flow

```
HTTP Request
    ↓
Uvicorn (ASGI server — like Tomcat/Netty)
    ↓
ASGI protocol (the contract — like Servlet API)
    ↓
FastAPI (framework — like Spring MVC/Express)
    ↓
Your endpoint function
```

## Why FastAPI has no built-in server (and why that's weird coming from Java/Node)

FastAPI is literally just the routing + validation + serialization layer. It produces an ASGI-compatible app object but has **zero ability to listen on a port**. It's like writing Spring MVC controllers without Tomcat — you have the handlers but nothing to receive requests.

### Python separates concerns more aggressively than other ecosystems

```
Java:    spring-boot-starter-web = Tomcat + Spring MVC + Jackson (all-in-one)
Node:    express = sits on top of Node's built-in http (runtime has it)
Python:  fastapi = JUST the framework. Server? Pick one yourself.
```

### What happens if you install FastAPI without Uvicorn?

```bash
pip install fastapi
python -c "import fastapi; print('works')"  # installs fine
uvicorn main:app  # command not found — you literally can't run your app
```

Your app is just a Python object sitting in memory. Nobody is listening on port 8000.

**The closest Java analogy:** Imagine if Spring MVC shipped without Tomcat, and you had to `mvn install tomcat` separately, then run `tomcat --deploy spring-app.war`. That's exactly what Python does.

### Why they did this — swappable servers

| Server | Use case |
|--------|----------|
| Uvicorn | Fast, single process, development + production |
| Gunicorn + Uvicorn workers | Production multi-process (like running multiple Tomcat threads) |
| Hypercorn | HTTP/2 support |
| Daphne | Django Channels / WebSockets |

Same app, different servers. In Spring, switching from Tomcat to Jetty requires changing dependencies and config. In Python, you just run a different command.

**In practice though**, `pip install "fastapi[standard]"` installs Uvicorn automatically. They know the separation is annoying so they bundle it as an optional dependency. But architecturally, they are two completely independent packages.

## Key takeaway

In Java, the server is hidden inside Spring Boot. In Node, the runtime IS the server. In Python, **you explicitly choose and run the server**. Uvicorn is the most popular choice for FastAPI because it's fast (built on `uvloop`, a C-based event loop — similar to how Node uses `libuv`).
