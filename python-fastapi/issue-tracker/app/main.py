# Exercise 01: Hello World
# Goal: Create a basic FastAPI app with a single endpoint
#
# Run with: uvicorn main:app --reload
# Then visit: http://127.0.0.1:8000
# API docs: http://127.0.0.1:8000/docs

# YOUR CODE HERE

from fastapi import FastAPI

app = FastAPI()

items = [
    {"id": 1, "name": "Item Number One"},
    {"id": 2, "name": "Item Two"},
    {"id": 3, "name": "Item Three"},
    {"id": 4, "name": "Item Four"}
]

@app.get("/health")       # decorator
async def health_check():
    return {"status": "ok"}

@app.get("/items/{item_id}")
def get_items(item_id: int):
    for item in items:
        if item["id"] == item_id:
            return item
    return {"error": "Item not found!"}

@app.post("/items")
def create_item(item: dict):
    items.append(item)
    return item