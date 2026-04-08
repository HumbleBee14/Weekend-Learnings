# Pydantic Validators — field_validator, model_validator, custom

## Validation order

Understanding when each validator runs is critical:

```
Raw input arrives
    ↓
1. Type coercion          "25" → 25 (str to int)
    ↓
2. Field constraints      Field(min_length=2, ge=18) — built-in checks
    ↓
3. @field_validator       custom logic per individual field
    ↓
4. @model_validator       custom logic across multiple fields
    ↓
Object created ✓
```

## 1. Built-in Field constraints (no custom code needed)

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(min_length=2, max_length=100)      # @Size(min=2, max=100)
    price: float = Field(gt=0)                            # @Positive — must be > 0
    quantity: int = Field(ge=0, le=10000)                 # @Min(0) @Max(10000)
    sku: str = Field(pattern=r'^[A-Z]{3}-\d{4}$')        # @Pattern — e.g., "ABC-1234"
    category: str = Field(default="general")              # optional with default
    tags: list[str] = Field(default_factory=list)         # optional, defaults to empty list
```

### All Field options

| Field option | What it does | Java equivalent |
|---|---|---|
| `min_length` / `max_length` | String length | `@Size(min=, max=)` |
| `gt` / `ge` / `lt` / `le` | Greater than / greater-equal / less than / less-equal | `@Min` / `@Max` / `@Positive` |
| `pattern` | Regex match | `@Pattern(regexp=)` |
| `default` | Default value | Field initializer |
| `default_factory` | Default from function (for mutable types) | No direct equivalent |
| `alias` | Accept different key name in JSON | `@JsonProperty("other_name")` |
| `exclude` | Exclude from serialization | `@JsonIgnore` |

## 2. @field_validator — custom logic for ONE field

Use when built-in Field constraints aren't enough.

```python
from pydantic import BaseModel, field_validator

class CreateUserRequest(BaseModel):
    name: str
    email: str
    username: str

    # Basic: strip whitespace and validate
    @field_validator('name')
    @classmethod
    def name_must_not_be_blank(cls, v):
        if v.strip() == '':
            raise ValueError('name cannot be blank')
        return v.strip()    # MUST return value — can transform it

    # Validate email format
    @field_validator('email')
    @classmethod
    def email_must_be_valid(cls, v):
        if '@' not in v:
            raise ValueError('invalid email address')
        return v.lower()    # normalize to lowercase

    # Validate multiple fields with SAME logic — pass multiple field names
    @field_validator('name', 'username')
    @classmethod
    def no_special_characters(cls, v):
        if not v.replace(' ', '').isalnum():
            raise ValueError('must contain only letters, numbers, and spaces')
        return v
```

## Understanding mode='before' vs mode='after'

Think of Pydantic as an assembly line. Your raw input goes through stages:

```
Raw JSON input: {"price": "$25.50", "name": "  alice  ", "age": "30"}
    │
    ▼
┌─ STAGE 1: mode='before' validators run HERE ─────────────────┐
│  Raw data, no type checking yet                                │
│  price is still "$25.50" (string with $)                       │
│  age is still "30" (string, not int)                           │
│  You can reshape, rename, preprocess anything                  │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ STAGE 2: Type coercion + Field constraints ─────────────────┐
│  "$25.50" → would FAIL converting to float (has $)             │
│  "30" → 30 (auto-coerced to int)                               │
│  Field(min_length=2) checked                                   │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ STAGE 3: mode='after' validators run HERE ──────────────────┐
│  All types are correct, all Fields passed                      │
│  price is float 25.50                                          │
│  age is int 30                                                 │
│  name is str "  alice  "                                       │
│  You validate business logic, transform values                 │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
Object created ✓
```

### Why mode='before' exists — without it, this fails

```python
class Product(BaseModel):
    price: float    # Pydantic tries to convert "$25.50" → float → CRASHES

# Input: {"price": "$25.50"}
# Error: "Input should be a valid number"
# Your mode='after' validator NEVER runs — type coercion failed first
```

With `mode='before'`, you intercept BEFORE Pydantic touches the type:

```python
class Product(BaseModel):
    price: float

    @field_validator('price', mode='before')
    @classmethod
    def strip_currency(cls, v):
        if isinstance(v, str):
            return v.replace('$', '').replace('₹', '')    # "$25.50" → "25.50"
        return v
    # NOW Pydantic converts "25.50" → float 25.50 → success

    @field_validator('price')    # mode='after' is the default
    @classmethod
    def must_be_positive(cls, v):
        # v is already a float here — safe to do math
        if v <= 0:
            raise ValueError('price must be positive')
        return v
```

### The Java analogy for before/after

```java
// mode='before' = custom Jackson @JsonDeserializer
// Runs BEFORE Jackson converts JSON string to Java type
public class PriceDeserializer extends JsonDeserializer<Double> {
    public Double deserialize(JsonParser p, DeserializationContext ctx) {
        String raw = p.getValueAsString();                    // "$25.50"
        return Double.parseDouble(raw.replace("$", ""));      // 25.50
    }
}

// mode='after' = Bean Validation @Constraint
// Runs AFTER Jackson has already created the typed object
@Positive    // validates that the double is > 0
private double price;
```

In Spring, these are two completely different systems (Jackson vs Bean Validation). In Pydantic, it's just `mode='before'` vs `mode='after'` on the same validator.

### When to use which

| | mode='before' | mode='after' (default) |
|---|---|---|
| **Data you receive** | Raw input — could be anything | Typed, validated values |
| **First param** | `cls` (no object yet) | `cls` for field_validator, `self` for model_validator |
| **Use when** | Need to clean/reshape raw input before Pydantic parses it | Need to validate business rules on clean data |
| **Examples** | Strip currency symbols, rename legacy keys, normalize formats | Check positive price, passwords match, end > start date |
| **Fails safely?** | If you don't handle it, type coercion may crash | Type coercion already passed, you're working with safe types |

**Rule of thumb:** if your validator needs to **fix the data so Pydantic can parse it** → `mode='before'`. If your validator needs to **check the parsed data makes business sense** → `mode='after'`.

### @field_validator with mode='before' — example

```python
class Product(BaseModel):
    price: float

    # Default mode='after' — runs after "25.5" is already converted to float 25.5
    @field_validator('price')
    @classmethod
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('price must be positive')
        return v

    # mode='before' — runs on RAW input, before any type conversion
    @field_validator('price', mode='before')
    @classmethod
    def strip_currency_symbol(cls, v):
        if isinstance(v, str):
            return v.replace('$', '').replace('₹', '').strip()
        return v
    # Now accepts: {"price": "$25.50"} → strips "$" → Pydantic converts "25.50" to float
```

## 3. @model_validator — custom logic across MULTIPLE fields

Use when validation depends on the relationship between fields. Same before/after concept applies here.

### mode='after' — all fields are already validated, object exists

```python
from pydantic import BaseModel, model_validator

class DateRange(BaseModel):
    start_date: str
    end_date: str

    @model_validator(mode='after')
    def end_must_be_after_start(self):      # NOTE: "self" not "cls" — object exists now
        if self.end_date <= self.start_date:
            raise ValueError('end_date must be after start_date')
        return self                          # MUST return self
```

### mode='before' — raw dict, before any field validation

```python
class PaymentRequest(BaseModel):
    amount: float
    currency: str
    amount_in_cents: int | None = None

    @model_validator(mode='before')
    @classmethod
    def convert_cents_to_amount(cls, data):   # data is raw dict, not an object
        if 'amount_in_cents' in data and 'amount' not in data:
            data['amount'] = data['amount_in_cents'] / 100
        return data                            # MUST return the dict
```

### Real-world: handling legacy API formats with mode='before'

```python
class Order(BaseModel):
    items: list[str]
    total: float
    discount_code: str | None = None

    # mode='before' — reshape raw input before Pydantic sees it
    @model_validator(mode='before')
    @classmethod
    def handle_legacy_format(cls, data):
        # Old API sent "item_list" instead of "items"
        if 'item_list' in data and 'items' not in data:
            data['items'] = data.pop('item_list')     # rename the key
        return data    # return modified dict → Pydantic continues with this

    # mode='after' — business logic on validated object
    @model_validator(mode='after')
    def validate_total(self):
        if len(self.items) > 0 and self.total <= 0:
            raise ValueError('order with items must have positive total')
        return self
```

```java
// Java equivalent — you'd need a custom deserializer or @JsonCreator:
// @JsonCreator
// public PaymentRequest(@JsonProperty("amount") Double amount,
//                       @JsonProperty("amount_in_cents") Integer cents) {
//     this.amount = amount != null ? amount : cents / 100.0;
// }
```

### Real-world example: password confirmation

```python
class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)
    password_confirm: str

    @model_validator(mode='after')
    def passwords_must_match(self):
        if self.password != self.password_confirm:
            raise ValueError('passwords do not match')
        return self
```

In Spring, you'd write a class-level `@Constraint` annotation + a `ConstraintValidator` implementation class. That's ~30 lines for what Pydantic does in 5.

## When to use which

```
Need to validate ONE field?
    ↓
Can Field() handle it? (min/max/pattern/gt/lt)
    ├── YES → use Field()
    └── NO  → use @field_validator
              Need to transform the raw input before type checking?
              → use mode='before'

Need to validate ACROSS fields?
    ↓
Use @model_validator
    ├── Need the validated object → mode='after' (self)
    └── Need to preprocess raw dict → mode='before' (cls, data)
```

## Summary — @classmethod vs self in validators

| Validator | Decorator | First param | Why |
|---|---|---|---|
| `@field_validator` | `@classmethod` | `cls` | Object doesn't exist yet — validating individual fields before construction |
| `@model_validator(mode='before')` | `@classmethod` | `cls` | Working with raw dict — object doesn't exist yet |
| `@model_validator(mode='after')` | None | `self` | Object is fully constructed — can access all fields via self |

The rule: **if the object exists, you get `self`. If it doesn't exist yet, you get `cls`.**
