# Decorators

## Java annotations vs Python decorators

They look similar (`@something`) but work completely differently.

**Java annotation = metadata.** It does nothing by itself. The Spring framework reads it at runtime via reflection and decides what to do.

**Python decorator = a function that wraps another function.** It actually executes at import time and replaces your function.

```python
# What you write:
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# What Python actually does (desugared):
async def health_check():
    return {"status": "ok"}
health_check = app.get("/health")(health_check)  # your function is passed INTO app.get()
```

So `app.get("/health")` returns a wrapper function, which receives `health_check` as an argument, registers the route internally, and returns the (possibly wrapped) function back. It's real code execution, not metadata.

### Build your own to see it clearly

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("BEFORE the function runs")
        result = func(*args, **kwargs)
        print("AFTER the function runs")
        return result
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
# Output:
# BEFORE the function runs
# Hello!
# AFTER the function runs
```

### The mental model

```
Java:    @GetMapping("/health")  → "hey Spring, when you scan this class, register this route"
                                    (annotation = sticky note for the framework to read later)

Python:  @app.get("/health")     → "right now, execute app.get(), pass my function into it, register it"
                                    (decorator = function call that happens immediately)
```
