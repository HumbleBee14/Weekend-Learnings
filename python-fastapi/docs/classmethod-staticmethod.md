# @classmethod vs @staticmethod vs instance method

## The three types of methods in Python

```python
class User:
    company = "Acme Corp"    # class variable (shared across all instances — like static field in Java)

    def __init__(self, name: str):
        self.name = name     # instance variable

    # 1. Instance method — has access to the instance (self)
    def greet(self):
        return f"Hi, I'm {self.name} from {self.company}"

    # 2. Class method — has access to the class (cls), NOT an instance
    @classmethod
    def from_email(cls, email: str):
        name = email.split("@")[0]
        return cls(name)     # cls = User class, so this calls User(name)

    # 3. Static method — has access to NOTHING, just lives inside the class
    @staticmethod
    def is_valid_email(email: str) -> bool:
        return "@" in email and "." in email
```

## The Java mapping

```java
public class User {
    static String company = "Acme Corp";    // class variable
    private String name;                     // instance variable

    // Instance method — same as Python's def method(self)
    public String greet() {
        return "Hi, I'm " + this.name + " from " + company;
    }

    // Factory method — closest to Python's @classmethod
    public static User fromEmail(String email) {
        String name = email.split("@")[0];
        return new User(name);
    }

    // Static utility — same as Python's @staticmethod
    public static boolean isValidEmail(String email) {
        return email.contains("@") && email.contains(".");
    }
}
```

## Side-by-side comparison

| | Python | Java | Has access to |
|---|---|---|---|
| Instance method | `def greet(self)` | `public void greet()` | `self` / `this` — the object |
| Class method | `@classmethod def from_email(cls)` | `public static User fromEmail()` | `cls` — the class itself |
| Static method | `@staticmethod def is_valid()` | `public static boolean isValid()` | Nothing — just a function |

## Why does @classmethod exist separately from @staticmethod?

In Java, both would just be `static` methods. Python splits them because **`cls` enables inheritance-aware factory methods**:

```python
class User(BaseModel):
    name: str

    @classmethod
    def guest(cls):
        return cls(name="Guest")    # cls = whatever class calls this

class AdminUser(User):
    role: str = "admin"

user = User.guest()          # cls = User → returns User(name="Guest")
admin = AdminUser.guest()    # cls = AdminUser → returns AdminUser(name="Guest", role="admin")
```

If this was `@staticmethod`, you'd have to hardcode `User(name="Guest")` — it wouldn't know about `AdminUser`. `cls` makes it polymorphic.

```java
// Java can't do this with static methods — static methods don't participate in polymorphism
// You'd need the abstract factory pattern or generics to achieve the same thing
```

## Why Pydantic's @field_validator requires @classmethod

Validation runs **before the object is created** — there's no `self` yet:

```python
class CreateUserRequest(BaseModel):
    name: str

    @field_validator('name')
    @classmethod                    # required — object doesn't exist yet, only the class
    def validate_name(cls, v):      # cls = CreateUserRequest class, v = raw value
        return v.strip()

# Timeline:
# 1. Raw data arrives: {"name": "  Alice  "}
# 2. Pydantic calls CreateUserRequest.validate_name("  Alice  ")
#    → class method, no instance exists yet
# 3. Returns "Alice"
# 4. NOW the object is created with name="Alice"
```

In Java terms: it's like running validation logic in a static factory method before calling the constructor.

## When to use which

| Use | When |
|---|---|
| Instance method (`self`) | Needs to read/modify the object's data |
| `@classmethod` (`cls`) | Factory methods, alternative constructors, Pydantic validators |
| `@staticmethod` | Pure utility — doesn't need the class or instance, but logically belongs to the class |

### Rule of thumb

If you'd make it a `static` method in Java, ask: "Does this need to know which class called it?"
- **Yes** → `@classmethod`
- **No** → `@staticmethod`
