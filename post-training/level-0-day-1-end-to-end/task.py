"""
The task: turn a messy employee record into strict JSON.

This one module has three jobs, on purpose — remember "data & algorithm co-design":
  1. render_row(record)   -> a messy, natural-language line   (used to GENERATE data)
  2. build_prompt(row)    -> the instruction the model sees
  3. score(output, gold)  -> the VERIFIER. It is our eval metric now, and it
                             becomes the GRPO *reward function* in Level 4.
                             One verifiable checker, three roles.

Pure Python. No GPU, no ML library. Runs anywhere, including your Mac.
"""

from __future__ import annotations
import json
import random
import re

# The schema we are teaching the model to emit.
FIELDS = ["name", "age", "department", "start_date", "salary"]
DEPARTMENTS = ["Sales", "Engineering", "Marketing", "Finance", "Support", "HR"]

_FIRST = ["Ava", "Liam", "Noah", "Mia", "Raj", "Sana", "Chen", "Omar",
          "Elena", "Kofi", "Yuki", "Diego", "Aisha", "Tom", "Priya", "Ivan"]
_LAST = ["Patel", "Kim", "Garcia", "Okafor", "Nguyen", "Smith", "Rossi",
         "Haddad", "Silva", "Novak", "Tanaka", "Mensah", "Cohen", "Ali"]

INSTRUCTION = (
    "Extract the employee record below into a JSON object with exactly these keys: "
    "name (string), age (integer), "
    "department (one of: Sales, Engineering, Marketing, Finance, Support, HR), "
    "start_date (string, normalized to YYYY-MM-DD), salary (integer, in dollars). "
    "Reply with ONLY the JSON object and nothing else."
)


def make_record(rng: random.Random) -> dict:
    """A ground-truth record — the canonical answer."""
    name = f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
    year, month, day = rng.randint(2015, 2024), rng.randint(1, 12), rng.randint(1, 28)
    return {
        "name": name,
        "age": rng.randint(22, 60),
        "department": rng.choice(DEPARTMENTS),
        "start_date": f"{year:04d}-{month:02d}-{day:02d}",
        "salary": rng.randint(40, 200) * 1000,
    }


# Several messy surface forms. Same facts, different order / separators / phrasing.
# This variety is what forces the model to LEARN THE MAPPING, not memorize a format.
_TEMPLATES = [
    "Name: {name}, Age {age}, Dept={department}, Started {start_date}, Salary {salary}",
    "{name} ({age}) - {department} team, joined {start_date}, earns {salary}/yr",
    "employee {name}; department: {department}; age {age}; start date {start_date}; comp {salary}",
    "{name}\t{age}\t{department}\t{start_date}\t{salary}",
    "[{department}] {name}, aged {age}, on payroll since {start_date} at {salary}",
    "row -> {name} | {salary} | {department} | age {age} | start {start_date}",
]

# The crucial part: the INPUT shows each field in a messy human form the model
# must TRANSFORM, not copy. Copying is easy — a base model can do it. Normalizing
# "Aug 28, '23" -> "2023-08-28" is the skill SFT has to actually teach.

_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]

# Department synonyms as they appear in the wild. Gold stays canonical.
_DEPT_ALIASES = {
    "Sales": ["sales", "SALES", "biz dev", "Sales & Partnerships"],
    "Engineering": ["eng", "engg", "R&D", "SWE org", "ENGINEERING"],
    "Marketing": ["mktg", "marketing", "Growth/Marketing", "MKTG"],
    "Finance": ["fin", "finance dept", "FIN", "Accounts & Finance"],
    "Support": ["cust support", "CS team", "helpdesk", "SUPPORT"],
    "HR": ["human resources", "People Ops", "people team", "hr"],
}


def _messy_date(iso: str, rng: random.Random) -> str:
    """'2023-08-28' -> one of many human formats the model must normalize back."""
    y, m, d = int(iso[:4]), int(iso[5:7]), int(iso[8:10])
    month = _MONTHS[m - 1]
    forms = [
        f"{month} {d}, {y}",                      # August 28, 2023
        f"{d} {month[:3]} {y}",                   # 28 Aug 2023
        f"{month[:3]} {d}, '{str(y)[2:]}",        # Aug 28, '23
        f"{d:02d}/{m:02d}/{y}",                   # 28/08/2023 (day-first!)
        f"{m}/{d}/{str(y)[2:]}",                  # 8/28/23 (US short)
        f"{d}th of {month}, {y}" if 4 <= d <= 20 else f"{d} of {month}, {y}",
    ]
    return rng.choice(forms)


def _messy_salary(amount: int, rng: random.Random) -> str:
    """75000 -> '75k' / '$75,000 USD' / '75,000 per year' — must come back as 75000."""
    k = amount // 1000
    forms = [
        f"{k}k", f"${k}K", f"${amount:,}", f"{amount:,} USD",
        f"${amount:,} per year", f"USD {amount:,}", f"{k}k/yr",
    ]
    return rng.choice(forms)


def _messy_age(age: int, rng: random.Random) -> str:
    """Age occasionally spelled out in words or with noise around it."""
    forms = [str(age), str(age), str(age), f"{age} yrs", f"{age} years old"]
    return rng.choice(forms)


def render_row(record: dict, rng: random.Random) -> str:
    """Render a ground-truth record as one messy input line requiring NORMALIZATION."""
    messy = {
        "name": record["name"],
        "age": _messy_age(record["age"], rng),
        "department": rng.choice(_DEPT_ALIASES[record["department"]]),
        "start_date": _messy_date(record["start_date"], rng),
        "salary": _messy_salary(record["salary"], rng),
    }
    return rng.choice(_TEMPLATES).format(**messy)


def build_prompt(row: str) -> str:
    """The full prompt (raw text, no chat template — identical at train and eval time)."""
    return f"{INSTRUCTION}\n\nRecord: {row}\nJSON:"


def target_completion(record: dict) -> str:
    """The gold JSON string, canonical key order."""
    return json.dumps({k: record[k] for k in FIELDS}, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# The verifier — eval metric now, RL reward later.
# --------------------------------------------------------------------------- #

def extract_json(text: str):
    """Best-effort: pull the first JSON object out of a model's raw output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _norm(field: str, value) -> str:
    """Normalize a field value so trivial formatting differences don't count as wrong."""
    if field in ("age", "salary"):
        try:
            return str(int(str(value).replace(",", "").replace("$", "").strip()))
        except Exception:
            return str(value).strip().lower()
    return str(value).strip().lower()


def score(output: str, gold: dict) -> dict:
    """
    Grade one model output against the gold record.

    Returns:
      parse_ok        : did we get valid JSON at all?
      field_accuracy  : fraction of the 5 fields that match (0.0 if unparsed)
      exact_match     : all fields correct?
      reward          : scalar in [0, 1] used as the RL reward in Level 4 (== field_accuracy)
    """
    pred = extract_json(output)
    if not isinstance(pred, dict):
        return {"parse_ok": False, "field_accuracy": 0.0, "exact_match": False, "reward": 0.0}
    correct = sum(1 for f in FIELDS if f in pred and _norm(f, pred[f]) == _norm(f, gold[f]))
    acc = correct / len(FIELDS)
    return {"parse_ok": True, "field_accuracy": acc,
            "exact_match": correct == len(FIELDS), "reward": acc}


if __name__ == "__main__":
    # Quick self-test: generate one example and score a perfect vs broken answer.
    rng = random.Random(0)
    rec = make_record(rng)
    row = render_row(rec, rng)
    print("row     :", row)
    print("prompt  :", build_prompt(row))
    print("gold    :", target_completion(rec))
    print("perfect :", score(target_completion(rec), rec))
    print("chatty  :", score("Sure! Here is the JSON:\n```json\n" + target_completion(rec) + "\n```", rec))
    print("broken  :", score("the employee is called " + rec["name"], rec))
