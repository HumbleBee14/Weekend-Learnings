"""
Locust file for load-testing the topic-03 batched server.

Run:
    pip install locust
    locust -f locustfile.py --host http://localhost:8000

Then open http://localhost:8089 — set 16 users, spawn rate 4/sec, click start.

For headless CSV output (the right way to capture G2):
    locust -f locustfile.py --host http://localhost:8000 \\
        --headless --users 16 --spawn-rate 4 --run-time 5m --csv g2
"""

import random

from locust import HttpUser, task, between

PROMPTS = [
    # Short prompts (~10-30 tokens)
    "Define recursion in two sentences.",
    "What is a hash table?",
    "Explain TCP vs UDP briefly.",
    "What is virtual memory?",
    "What does 'idempotent' mean?",
    # Medium prompts (~50-100 tokens)
    "Walk me through how a CPU cache hierarchy works, including L1, L2, and L3 caches and "
    "what happens on a cache miss.",
    "Explain how the operating system implements virtual memory using page tables and the TLB. "
    "Mention the role of the MMU.",
    # Longer prompts (~200+ tokens, simulating chat history)
    "I'm trying to understand databases. Walk me through what happens when I run a SELECT query "
    "with a JOIN. Cover query parsing, planning, optimization, the role of indexes, and how the "
    "buffer pool participates. Use a concrete example with two tables, customers and orders, and "
    "a WHERE clause that filters on a date range. I'm a senior engineer so don't oversimplify.",
]


class LLMUser(HttpUser):
    """
    One simulated user. Sends a request, waits 1-3 seconds, repeats.

    `wait_time = between(1, 3)` means the user pauses 1-3 seconds randomly between requests —
    roughly Poisson with rate ~0.5 req/s/user. With 16 users, expected QPS ≈ 8.
    """

    wait_time = between(1, 3)

    @task(3)  # weight 3 — runs 3x as often as the long_request task
    def short_request(self):
        prompt = random.choice(PROMPTS[:5])
        self.client.post(
            "/generate",
            json={"prompt": prompt, "max_tokens": 40, "temperature": 0.7},
            name="generate (short)",  # group all short prompts under one Locust label
        )

    @task(1)
    def long_request(self):
        prompt = random.choice(PROMPTS[5:])
        self.client.post(
            "/generate",
            json={"prompt": prompt, "max_tokens": 100, "temperature": 0.7},
            name="generate (long)",
        )
