# Pydantic & BaseModel

## What is BaseModel?

BaseModel is a **class** — you inherit from it. Not a decorator, not an annotation, not magic. Your class extends it (`class User(BaseModel)`) and gets validation, serialization, and type coercion through inheritance.

```python
from pydantic import BaseModel

class User(BaseModel):     # ← inheritance, like "extends" in Java
    name: str
    email: str
    age: int
```

### The Java equivalent

```java
// In Java you need: a class + annotations + getters/setters + Jackson + @Valid
// Pydantic collapses ALL of this into just the class definition

public class User {
    @NotNull private String name;
    @NotNull @Email private String email;
    @NotNull @Min(0) private int age;
    // + constructor, getters, setters (or Lombok @Data)
}
```

```python
# In Python — the type hint IS the validation
class User(BaseModel):
    name: str      # required, must be string — that's it
    email: str
    age: int
```

## What BaseModel gives you for free

```python
class User(BaseModel):
    name: str
    email: str
    age: int
    is_active: bool = True     # optional with default

# 1. Validation — rejects bad data
user = User(name="Alice", email="alice@test.com", age="not_a_number")
# → ValidationError: age - Input should be a valid integer

# 2. Auto-coercion — smart type conversion
user = User(name="Alice", email="alice@test.com", age="25")
# → age becomes int 25, not string "25"

# 3. Serialization — .model_dump() is like Jackson's objectMapper.writeValueAsString()
user = User(name="Alice", email="alice@test.com", age=25)
user.model_dump()
# → {"name": "Alice", "email": "alice@test.com", "age": 25, "is_active": True}

# 4. Attribute access — not dict fiddling
print(user.name)   # "Alice" — not user["name"]
```

## Why we pass BaseModel in method args

When FastAPI sees a BaseModel type hint in your endpoint, it automatically:
1. Reads the JSON request body
2. Validates it against the model
3. Returns 422 with detailed errors if invalid
4. Passes the validated, typed object to your function
5. Generates Swagger/OpenAPI docs from the model

```python
@app.post("/users")
async def create_user(user: CreateUserRequest):   # ← FastAPI sees BaseModel type hint
    # "user" is already validated, parsed, and typed
    # if someone sends bad JSON, they get a 422 BEFORE your code even runs
    print(user.name)   # guaranteed to be a string
    print(user.age)    # guaranteed to be an int
```

In Spring you'd write `@RequestBody @Valid CreateUserRequest user`. The `@RequestBody` tells Spring to parse JSON. The `@Valid` tells Spring to validate. In FastAPI, just the type hint does both.

## How FastAPI uses it — request and response models

```python
class CreateUserRequest(BaseModel):    # what the client sends
    name: str
    email: str
    age: int

class UserResponse(BaseModel):         # what the client receives
    id: int
    name: str
    email: str

@app.post("/users", response_model=UserResponse)
async def create_user(user: CreateUserRequest):
    # response_model strips out any extra fields — only id, name, email go back
    # like Jackson's @JsonIgnore but declarative at the endpoint level
    return UserResponse(id=1, name=user.name, email=user.email)
```

## The Java mapping

| Pydantic (Python) | Spring/Java |
|---|---|
| `BaseModel` | DTO class + `@Valid` + Jackson |
| Type hints (`name: str`) | `@NotNull private String name` + getter/setter |
| `Field(min_length=2)` | `@Size(min = 2)` |
| `.model_dump()` | `objectMapper.writeValueAsString()` |
| Auto-coercion (`"25"` → `25`) | Jackson deserialization |
| `response_model=UserResponse` | `@JsonView` or separate response DTO |
| Validation errors → auto 422 | `@ControllerAdvice` + exception handler |

## Validation errors — fully automatic

This is where FastAPI shines over Spring. Validation errors are handled **automatically**. You write zero error handling code.

### What happens when validation fails

```python
class CreateUserRequest(BaseModel):
    name: str
    email: str
    age: int

@app.post("/users")
async def create_user(user: CreateUserRequest):
    return {"name": user.name}
```

Client sends bad JSON:
```json
{"name": 123, "age": "old"}
```

FastAPI auto-responds with **422 Unprocessable Entity**:
```json
{
  "detail": [
    {
      "loc": ["body", "email"],
      "msg": "Field required",
      "type": "missing"
    },
    {
      "loc": ["body", "age"],
      "msg": "Input should be a valid integer, unable to parse string as an integer",
      "type": "int_parsing"
    }
  ]
}
```

Note: `name: 123` did NOT error — Pydantic auto-coerced int `123` to string `"123"`. Smart.

### The flow

```
Client sends JSON
    ↓
FastAPI reads the request body
    ↓
Pydantic validates against the BaseModel
    ↓
  ┌─── Valid? ──────────────────────────────── Your endpoint code runs
  │
  └─── Invalid? ────────────────────────────── FastAPI returns 422 automatically
                                                (your code NEVER executes)
```

### In Spring, you'd need all of this for the same behavior:

```java
// 1. The DTO with validation annotations
public class CreateUserRequest {
    @NotNull private String name;
    @NotNull @Email private String email;
    @NotNull @Min(0) private int age;
}

// 2. The controller with @Valid
@PostMapping("/users")
public ResponseEntity<?> createUser(@RequestBody @Valid CreateUserRequest user) { ... }

// 3. A global exception handler (without this, Spring returns ugly 500 errors)
@ControllerAdvice
public class ValidationHandler {
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<Map<String, Object>> handleValidation(MethodArgumentNotValidException ex) {
        List<Map<String, String>> errors = ex.getBindingResult()
            .getFieldErrors()
            .stream()
            .map(e -> Map.of("field", e.getField(), "message", e.getDefaultMessage()))
            .toList();
        return ResponseEntity.status(422).body(Map.of("detail", errors));
    }
}
```

In FastAPI? Just define the BaseModel. That's it. The 422 response with field-level errors is built in.

### Custom error responses (if you want to override)

You CAN customize error handling, but you don't have to:

```python
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def custom_validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,    # use 400 instead of 422
        content={"message": "Invalid input", "errors": exc.errors()}
    )
```

This is like `@ControllerAdvice` in Spring — but you only write it if you want to change the default behavior.

## Advanced validation with Field

```python
from pydantic import BaseModel, Field, field_validator

class CreateUserRequest(BaseModel):
    name: str = Field(min_length=2, max_length=50)          # like @Size(min=2, max=50)
    email: str = Field(pattern=r'^[\w.-]+@[\w.-]+\.\w+$')   # like @Email
    age: int = Field(ge=18, le=120)                          # like @Min(18) @Max(120)
    role: str = Field(default="user")                        # optional with default

    # Custom validator — like writing a custom @Constraint annotation in Java
    @field_validator('name')
    @classmethod
    def name_must_not_be_blank(cls, v):
        if v.strip() == '':
            raise ValueError('name cannot be blank')
        return v.strip()
```

## Nested models

```python
class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class User(BaseModel):
    name: str
    email: str
    address: Address                 # nested object — validated recursively
    tags: list[str] = []             # List<String> equivalent
    scores: dict[str, int] = {}     # Map<String, Integer> equivalent

# Accepts:
# {
#   "name": "Alice",
#   "email": "alice@test.com",
#   "address": {"street": "123 Main", "city": "Mumbai", "zip_code": "400001"},
#   "tags": ["admin", "active"],
#   "scores": {"math": 95, "science": 88}
# }
```

In Java you'd need `@Valid` on the nested `Address` field for validation to cascade. Pydantic does it automatically for any nested BaseModel.
