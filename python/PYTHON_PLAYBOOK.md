# Python Expert Playbook

Quick-reference guide for expert-level Python interviews. Each topic has a plain-English explanation followed by runnable code with inline comments showing output.

---

## Table of Contents

- [Python Expert Playbook](#python-expert-playbook)
  - [Table of Contents](#table-of-contents)
  - [1. Fundamentals \& Data Types](#1-fundamentals--data-types)
    - [Identity vs Equality](#identity-vs-equality)
    - [Mutability](#mutability)
    - [f-strings](#f-strings)
    - [Walrus Operator (`:=`)](#walrus-operator-)
    - [Comprehensions vs Generators](#comprehensions-vs-generators)
    - [Unpacking](#unpacking)
    - [LEGB Scope](#legb-scope)
    - [Truthiness \& Short-circuit](#truthiness--short-circuit)
  - [2. OOP Advanced](#2-oop-advanced)
    - [`__new__` vs `__init__`](#__new__-vs-__init__)
    - [Properties](#properties)
    - [MRO and `super()`](#mro-and-super)
    - [Abstract Base Classes (ABC)](#abstract-base-classes-abc)
    - [Dunder (Magic) Methods](#dunder-magic-methods)
    - [Descriptors](#descriptors)
    - [Metaclasses](#metaclasses)
    - [`__slots__`](#__slots__)
    - [Dataclasses](#dataclasses)
  - [3. Functional Programming](#3-functional-programming)
    - [Closures](#closures)
    - [Decorators](#decorators)
    - [Decorator with Arguments](#decorator-with-arguments)
    - [Generators](#generators)
    - [`yield from`](#yield-from)
    - [`itertools` Essentials](#itertools-essentials)
    - [`functools` Essentials](#functools-essentials)
  - [4. Concurrency](#4-concurrency)
    - [The GIL](#the-gil)
    - [Threading Lock](#threading-lock)
    - [Semaphore](#semaphore)
    - [`ThreadPoolExecutor`](#threadpoolexecutor)
    - [`asyncio` — Coroutines \& `gather`](#asyncio--coroutines--gather)
    - [`asyncio.Queue` — Producer-Consumer](#asyncioqueue--producer-consumer)
    - [`asyncio.to_thread` — Blocking Code in Async](#asyncioto_thread--blocking-code-in-async)
  - [5. Memory \& Performance](#5-memory--performance)
    - [Reference Counting](#reference-counting)
    - [Cyclic GC and `weakref`](#cyclic-gc-and-weakref)
    - [`sys.getsizeof` and `tracemalloc`](#sysgetsizeof-and-tracemalloc)
    - [Python Profiling](#python-profiling)
      - [How Profiling Works Internally](#how-profiling-works-internally)
      - [`timeit` — Benchmark a Snippet](#timeit--benchmark-a-snippet)
      - [`cProfile` — Function-Level Deterministic Profiler](#cprofile--function-level-deterministic-profiler)
      - [`line_profiler` — Line-by-Line Timing](#line_profiler--line-by-line-timing)
      - [`memory_profiler` — Line-by-Line Memory](#memory_profiler--line-by-line-memory)
      - [`tracemalloc` — Memory Snapshots (stdlib)](#tracemalloc--memory-snapshots-stdlib)
      - [`py-spy` — Statistical Profiler (Zero Overhead, Production-Safe)](#py-spy--statistical-profiler-zero-overhead-production-safe)
      - [Profiling and Parallelism](#profiling-and-parallelism)
      - [Profiling Tool Cheat Sheet](#profiling-tool-cheat-sheet)
      - [Profiling Workflow](#profiling-workflow)
    - [`__slots__` for Memory Savings](#__slots__-for-memory-savings)
    - [`array` and `memoryview`](#array-and-memoryview)
    - [`copy.copy` vs `copy.deepcopy`](#copycopy-vs-copydeepcopy)
  - [6. Data Structures \& Algorithms](#6-data-structures--algorithms)
    - [List vs Dict vs Set Complexity](#list-vs-dict-vs-set-complexity)
    - [`collections.deque`](#collectionsdeque)
    - [`heapq` — Priority Queue](#heapq--priority-queue)
    - [`bisect` — Binary Search](#bisect--binary-search)
    - [Floyd's Cycle Detection](#floyds-cycle-detection)
    - [Binary Tree Traversals](#binary-tree-traversals)
    - [Dynamic Programming](#dynamic-programming)
    - [Two Pointers \& Sliding Window](#two-pointers--sliding-window)
  - [7. Design Patterns](#7-design-patterns)
    - [Singleton](#singleton)
    - [Factory Pattern](#factory-pattern)
    - [Builder Pattern](#builder-pattern)
    - [Observer Pattern](#observer-pattern)
    - [Strategy Pattern](#strategy-pattern)
    - [Context Manager](#context-manager)
    - [Command Pattern (Undo/Redo)](#command-pattern-undoredo)
  - [8. Testing](#8-testing)
    - [`unittest` Lifecycle](#unittest-lifecycle)
    - [pytest — Plain Functions](#pytest--plain-functions)
    - [Fixtures](#fixtures)
    - [Parametrize](#parametrize)
    - [`Mock` and `MagicMock`](#mock-and-magicmock)
    - [`patch`](#patch)
    - [`create_autospec`](#create_autospec)
  - [9. Advanced Internals](#9-advanced-internals)
    - [Import System](#import-system)
    - [Bytecode and `dis`](#bytecode-and-dis)
    - [Frame Objects](#frame-objects)
    - [`exec` and `eval`](#exec-and-eval)
    - [`ast` Module](#ast-module)
    - [`weakref` — Prevent Memory Leaks in Caches](#weakref--prevent-memory-leaks-in-caches)
    - [`contextvars` — Per-Coroutine Storage](#contextvars--per-coroutine-storage)
    - [`__init_subclass__`](#__init_subclass__)
  - [10. Type System \& Protocols](#10-type-system--protocols)
    - [Type Hints Basics](#type-hints-basics)
    - [`TypeVar` and `Generic[T]`](#typevar-and-generict)
    - [`Protocol` — Structural Typing](#protocol--structural-typing)
    - [`TypedDict`](#typeddict)
    - [`Annotated` — Metadata in Type Hints](#annotated--metadata-in-type-hints)
    - [`@overload`](#overload)
    - [Pydantic v2 — Runtime Validation](#pydantic-v2--runtime-validation)
    - [`NewType` and `cast`](#newtype-and-cast)
    - [`TYPE_CHECKING` — Avoid Circular Imports](#type_checking--avoid-circular-imports)
  - [Quick Decision Guide](#quick-decision-guide)
  - [11. Processing Huge Data on Limited Memory](#11-processing-huge-data-on-limited-memory)
    - [The Core Problem](#the-core-problem)
    - [Strategy 1 — Generators for Zero-Copy Streaming](#strategy-1--generators-for-zero-copy-streaming)
    - [Strategy 2 — Chunked Processing + `ProcessPoolExecutor` (CPU-bound)](#strategy-2--chunked-processing--processpoolexecutor-cpu-bound)
    - [Strategy 3 — `ThreadPoolExecutor` (IO-bound, not CPU-bound)](#strategy-3--threadpoolexecutor-io-bound-not-cpu-bound)
    - [Strategy 4 — `asyncio` for Massive IO Concurrency](#strategy-4--asyncio-for-massive-io-concurrency)
    - [How asyncio Helps with Parallelism](#how-asyncio-helps-with-parallelism)
    - [Strategy 5 — `mmap` for Random Access Without Loading](#strategy-5--mmap-for-random-access-without-loading)
    - [Profiling to Find the Real Bottleneck First](#profiling-to-find-the-real-bottleneck-first)
    - [Decision Tree](#decision-tree)
    - [Quick Reference](#quick-reference)
   - [Type Hints Basics](#type-hints-basics)
   - [`TypeVar` and `Generic[T]`](#typevar-and-generict)
   - [`Protocol` — Structural Typing](#protocol-—-structural-typing)
   - [`TypedDict`](#typeddict)
   - [`Annotated` — Metadata in Type Hints](#annotated-—-metadata-in-type-hints)
   - [`@overload`](#overload)
   - [Pydantic v2 — Runtime Validation](#pydantic-v2-—-runtime-validation)
   - [`NewType` and `cast`](#newtype-and-cast)
   - [`TYPE_CHECKING` — Avoid Circular Imports](#type_checking-—-avoid-circular-imports)

11. [11. Processing Huge Data on Limited Memory](#11-processing-huge-data-on-limited-memory)
   - [The Core Problem](#the-core-problem)
   - [Strategy 1 — Generators for Zero-Copy Streaming](#strategy-1-—-generators-for-zero-copy-streaming)
   - [Strategy 2 — Chunked Processing + `ProcessPoolExecutor` (CPU-bound)](#strategy-2-—-chunked-processing--processpoolexecutor-cpu-bound)
   - [Strategy 3 — `ThreadPoolExecutor` (IO-bound)](#strategy-3-—-threadpoolexecutor-io-bound-not-cpu-bound)
   - [Strategy 4 — `asyncio` for Massive IO Concurrency](#strategy-4-—-asyncio-for-massive-io-concurrency)
   - [How asyncio Helps with Parallelism](#how-asyncio-helps-with-parallelism)
   - [Strategy 5 — `mmap` for Random Access](#strategy-5-—-mmap-for-random-access-without-loading)
   - [Profiling to Find the Real Bottleneck First](#profiling-to-find-the-real-bottleneck-first)
   - [Decision Tree](#decision-tree)
   - [Quick Reference](#quick-reference)

---

## 1. Fundamentals & Data Types

### Identity vs Equality

`==` checks if two values are **equal**. `is` checks if they are the **exact same object** in memory. CPython caches small integers (-5 to 256) and short strings, so `is` can surprise you.

```python
a = 256
b = 256
print(a == b)   # True  — same value
print(a is b)   # True  — same object (CPython caches small ints)

a = 257
b = 257
print(a is b)   # False — outside cache range, two separate objects

# Always use == for value comparison; use is only for None/True/False
print(None is None)   # True — correct idiom
```

---

### Mutability

**Mutable** objects can be changed after creation (list, dict, set).  
**Immutable** objects cannot (int, str, tuple). The classic trap: using a mutable default argument.

```python
# TRAP: mutable default argument is created ONCE, shared across all calls
def append_item(item, lst=[]):
    lst.append(item)
    return lst

print(append_item(1))   # [1]
print(append_item(2))   # [1, 2]  ← not what you expected!

# FIX: use None as default
def append_item_fixed(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst

print(append_item_fixed(1))  # [1]
print(append_item_fixed(2))  # [2]  ← fresh list each time
```

---

### f-strings

The fastest and most readable way to format strings in Python 3.6+. Supports expressions, format specs, and the `=` shorthand for debugging.

```python
name = "Alice"
score = 98.5

print(f"Hello, {name}!")           # Hello, Alice!
print(f"Score: {score:.1f}%")      # Score: 98.5%
print(f"{'hi'.upper()}")           # HI  — expressions work inside {}

# = shorthand: prints variable name + value (great for debugging)
x = 42
print(f"{x=}")     # x=42
print(f"{x*2=}")   # x*2=84
```

---

### Walrus Operator (`:=`)

Assigns a value **and** returns it in one step. Removes the need for a separate assignment before a condition.

```python
import re

# Without walrus — assign then check
data = "User: Alice"
match = re.search(r"User: (\w+)", data)
if match:
    print(match.group(1))   # Alice

# With walrus — assign inside the condition
if m := re.search(r"User: (\w+)", data):
    print(m.group(1))       # Alice

# Very useful in while loops reading chunks
# while chunk := file.read(8192):
#     process(chunk)
```

---

### Comprehensions vs Generators

**List comprehension** builds the full list in memory at once.  
**Generator expression** is lazy — computes one item at a time. Use generators for large data.

```python
# List comprehension — all values created immediately
squares = [x**2 for x in range(5)]
print(squares)         # [0, 1, 4, 9, 16]

# Generator expression — values created on demand (no [] brackets)
gen = (x**2 for x in range(5))
print(next(gen))       # 0
print(next(gen))       # 1

# Dict and set comprehensions
word_len = {w: len(w) for w in ["hi", "hello"]}
print(word_len)        # {'hi': 2, 'hello': 5}

evens = {x for x in range(10) if x % 2 == 0}
print(evens)           # {0, 2, 4, 6, 8}
```

---

### Unpacking

Python lets you unpack iterables into variables. The `*` collects the "rest" into a list.

```python
first, *rest = [1, 2, 3, 4, 5]
print(first)   # 1
print(rest)    # [2, 3, 4, 5]

a, b, *_, last = range(10)
print(a, b, last)   # 0 1 9

# Swap without temp variable
x, y = 10, 20
x, y = y, x
print(x, y)   # 20 10

# Unpack nested
(a, b), c = (1, 2), 3
print(a, b, c)   # 1 2 3
```

---

### LEGB Scope

Python looks up names in this order: **L**ocal → **E**nclosing → **G**lobal → **B**uilt-in.  
Use `global` to write to a global variable, `nonlocal` to write to an enclosing function's variable.

```python
x = "global"

def outer():
    x = "enclosing"

    def inner():
        x = "local"
        print(x)   # local — found in Local scope

    inner()
    print(x)       # enclosing

outer()
print(x)           # global

# nonlocal lets inner function write to enclosing scope
def make_counter():
    count = 0
    def inc():
        nonlocal count
        count += 1
        return count
    return inc

counter = make_counter()
print(counter())   # 1
print(counter())   # 2
```

---

### Truthiness & Short-circuit

Every object has a boolean value. Empty containers, 0, None, and "" are **falsy**. `and`/`or` return the **deciding value**, not True/False.

```python
# Falsy values
print(bool([]))      # False
print(bool({}))      # False
print(bool(0))       # False
print(bool(""))      # False
print(bool(None))    # False

# Truthy — everything else
print(bool([0]))     # True  — list with one item
print(bool("False")) # True  — non-empty string

# or returns first truthy value, and returns first falsy (or last)
name = "" or "Guest"
print(name)          # Guest

value = 0 or 42
print(value)         # 42

# Safe default pattern
config = None
port = config and config.get("port") or 8080
print(port)          # 8080
```


---

## 2. OOP Advanced

### `__new__` vs `__init__`

`__new__` **creates** the object (allocates memory). `__init__` **initializes** it (sets attributes). For normal classes you only need `__init__`. Override `__new__` when subclassing immutable types or implementing Singleton.

```python
class MyInt(int):
    def __new__(cls, value):
        print(f"Creating object with value {value}")
        return super().__new__(cls, value * 2)  # doubles the value at creation

    def __init__(self, value):
        print(f"Initializing (value stored is already {self})")

n = MyInt(5)
print(n)   # 10  — __new__ doubled it; __init__ can't change it (int is immutable)
```

---

### Properties

Use `@property` to make a method look like an attribute. Add `@name.setter` to allow writes with validation.

```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value

    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32   # computed — no setter needed

t = Temperature(25)
print(t.celsius)      # 25
print(t.fahrenheit)   # 77.0
t.celsius = 100
print(t.fahrenheit)   # 212.0
# t.celsius = -300    # raises ValueError
```

---

### MRO and `super()`

When multiple classes are inherited, Python uses **C3 linearization** to determine which method gets called first. `super()` always calls the **next class in MRO**, not just the parent.

```python
class A:
    def greet(self): print("A")

class B(A):
    def greet(self): print("B"); super().greet()

class C(A):
    def greet(self): print("C"); super().greet()

class D(B, C):
    def greet(self): print("D"); super().greet()

print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

D().greet()
# D
# B
# C
# A
```

---

### Abstract Base Classes (ABC)

ABCs define a **required interface** — subclasses must implement all `@abstractmethod` methods or Python raises an error at instantiation.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...

    @abstractmethod
    def perimeter(self) -> float: ...

    def describe(self):   # concrete method — shared by all subclasses
        return f"Area={self.area():.2f}, Perimeter={self.perimeter():.2f}"

class Circle(Shape):
    def __init__(self, r): self.r = r
    def area(self): return 3.14159 * self.r ** 2
    def perimeter(self): return 2 * 3.14159 * self.r

# Shape()      # TypeError: Can't instantiate abstract class
print(Circle(5).describe())   # Area=78.54, Perimeter=31.42
```

---

### Dunder (Magic) Methods

Special methods that Python calls automatically. They let your classes behave like built-in types.

```python
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __repr__(self):          # used in REPL and repr()
        return f"Vector({self.x}, {self.y})"

    def __add__(self, other):    # v1 + v2
        return Vector(self.x + other.x, self.y + other.y)

    def __len__(self):           # len(v)
        return 2

    def __eq__(self, other):     # v1 == v2
        return self.x == other.x and self.y == other.y

    def __abs__(self):           # abs(v)
        return (self.x**2 + self.y**2) ** 0.5

v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1 + v2)    # Vector(4, 6)
print(abs(v2))    # 5.0
print(v1 == Vector(1, 2))  # True
```

---

### Descriptors

A descriptor is a class that controls how an attribute is **read, set, or deleted** on another class. It's the mechanism behind `property`, `classmethod`, and `staticmethod`.

```python
class Validated:
    """Descriptor: ensures attribute is a positive number."""
    def __set_name__(self, owner, name):
        self.name = name          # auto-captures attribute name

    def __get__(self, obj, objtype=None):
        if obj is None: return self
        return obj.__dict__.get(self.name, 0)

    def __set__(self, obj, value):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{self.name} must be a positive number")
        obj.__dict__[self.name] = value

class Product:
    price = Validated()
    stock = Validated()

p = Product()
p.price = 9.99
p.stock = 100
print(p.price, p.stock)   # 9.99 100
# p.price = -1            # raises ValueError
```

---

### Metaclasses

A metaclass is the **class of a class**. When Python creates a class, it calls the metaclass. Use them to auto-register subclasses, enforce naming conventions, or add methods.

```python
class RegistryMeta(type):
    registry = {}

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if bases:   # skip the base class itself
            RegistryMeta.registry[name] = cls
        return cls

class Plugin(metaclass=RegistryMeta):
    pass

class CSVPlugin(Plugin): pass
class JSONPlugin(Plugin): pass

print(RegistryMeta.registry)
# {'CSVPlugin': <class 'CSVPlugin'>, 'JSONPlugin': <class 'JSONPlugin'>}
```

---

### `__slots__`

By default, each instance has a `__dict__` (a dictionary of its attributes). `__slots__` replaces this with a fixed array — saving ~200 bytes per instance and speeding up attribute access.

```python
class WithDict:
    def __init__(self, x, y):
        self.x, self.y = x, y

class WithSlots:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x, self.y = x, y

import sys
a = WithDict(1, 2)
b = WithSlots(1, 2)
print(sys.getsizeof(a.__dict__))   # ~232 bytes (the dict alone)
print(hasattr(b, '__dict__'))      # False — no dict!
# b.z = 3  # AttributeError — can't add new attributes
```

---

### Dataclasses

`@dataclass` auto-generates `__init__`, `__repr__`, and `__eq__`. Use `frozen=True` for immutability, `field()` for complex defaults.

```python
from dataclasses import dataclass, field
from typing import List

@dataclass(order=True, frozen=True)
class Point:
    x: float
    y: float

@dataclass
class Student:
    name: str
    grades: List[int] = field(default_factory=list)   # safe mutable default

    def average(self):
        return sum(self.grades) / len(self.grades) if self.grades else 0

p1, p2 = Point(1, 2), Point(3, 4)
print(p1 < p2)           # True  — order=True enables comparison
print(repr(p1))          # Point(x=1, y=2)

s = Student("Alice", [90, 85, 92])
print(s.average())       # 89.0
```


---

## 3. Functional Programming

### Closures

A closure is a function that **remembers variables from its enclosing scope** even after that scope has finished. It's the foundation of decorators and factory functions.

```python
def make_multiplier(factor):
    # 'factor' lives inside make_multiplier's scope
    def multiply(n):
        return n * factor   # multiply "closes over" factor
    return multiply

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))   # 10
print(triple(5))   # 15

# Inspect what a closure captures
print(double.__closure__[0].cell_contents)   # 2
```

---

### Decorators

A decorator is a function that **wraps another function** to add behaviour (logging, timing, auth). Always use `@functools.wraps` to preserve the original function's name and docstring.

```python
import functools, time

def timer(fn):
    @functools.wraps(fn)    # preserves fn.__name__, fn.__doc__
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[{fn.__name__}] took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_sum(n):
    return sum(range(n))

print(slow_sum(1_000_000))
# [slow_sum] took 0.0312s
# 499999500000
```

---

### Decorator with Arguments

When a decorator needs its own parameters, add an **extra outer function** that receives those params and returns the actual decorator.

```python
import functools

def retry(times=3, exceptions=(Exception,)):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    print(f"Attempt {attempt} failed: {e}")
                    if attempt == times:
                        raise
        return wrapper
    return decorator

@retry(times=3, exceptions=(ValueError,))
def risky(x):
    if x < 0:
        raise ValueError("negative!")
    return x * 2

print(risky(5))    # 10
# risky(-1)        # retries 3 times then raises ValueError
```

---

### Generators

A generator function uses `yield` to **pause and return a value**, then resume from where it left off. This makes it memory-efficient for large sequences.

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a       # pause here, return a
        a, b = b, a + b   # resume here on next()

gen = fibonacci()
print([next(gen) for _ in range(8)])   # [0, 1, 1, 2, 3, 5, 8, 13]

# Generator expression (like list comp but lazy)
total = sum(x**2 for x in range(1_000_000))   # no list created in memory
print(total)   # 333332833333500000
```

---

### `yield from`

`yield from` **delegates** to another iterable or generator — passes all values through, and also forwards `send()` / `throw()` calls.

```python
def flatten(nested):
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)   # recurse into sub-lists
        else:
            yield item

data = [1, [2, [3, 4]], [5, 6]]
print(list(flatten(data)))   # [1, 2, 3, 4, 5, 6]
```

---

### `itertools` Essentials

`itertools` provides fast, memory-efficient tools for working with iterables — no extra lists created.

```python
from itertools import chain, islice, groupby, product, accumulate

# chain — combine multiple iterables
print(list(chain([1, 2], [3, 4], [5])))   # [1, 2, 3, 4, 5]

# islice — lazy slice (works on infinite generators)
def naturals():
    n = 1
    while True: yield n; n += 1

print(list(islice(naturals(), 5)))   # [1, 2, 3, 4, 5]

# accumulate — running totals
print(list(accumulate([1, 2, 3, 4])))   # [1, 3, 6, 10]

# product — cartesian product
print(list(product("AB", [1, 2])))   # [('A',1),('A',2),('B',1),('B',2)]

# groupby — group consecutive same-key items (data must be sorted by key)
data = [("fruit","apple"),("fruit","banana"),("veg","carrot")]
for key, group in groupby(data, key=lambda x: x[0]):
    print(key, [v for _, v in group])
# fruit ['apple', 'banana']
# veg ['carrot']
```

---

### `functools` Essentials

Key tools for function manipulation and caching.

```python
import functools

# lru_cache — memoize expensive function calls
@functools.lru_cache(maxsize=128)
def fib(n):
    if n < 2: return n
    return fib(n-1) + fib(n-2)

print(fib(50))        # fast, even for large n
print(fib.cache_info())   # CacheInfo(hits=..., misses=51, ...)

# partial — pre-fill some arguments
def power(base, exp): return base ** exp
square = functools.partial(power, exp=2)
cube   = functools.partial(power, exp=3)
print(square(4), cube(3))   # 16 27

# singledispatch — overload by first argument type
@functools.singledispatch
def process(value):
    return f"unknown: {value}"

@process.register(int)
def _(value): return f"int: {value * 2}"

@process.register(str)
def _(value): return f"str: {value.upper()}"

print(process(5))       # int: 10
print(process("hi"))    # str: HI
```


---

## 4. Concurrency

### The GIL

The **Global Interpreter Lock** allows only one thread to execute Python bytecode at a time. For I/O-bound tasks (network, disk), the GIL is released while waiting — so threads help. For CPU-bound tasks, threads don't speed things up; use `multiprocessing` instead.

```python
import threading, time

def count_up(n):
    x = 0
    for _ in range(n): x += 1

# CPU-bound: two threads are NOT faster (GIL prevents true parallelism)
start = time.perf_counter()
t1 = threading.Thread(target=count_up, args=(5_000_000,))
t2 = threading.Thread(target=count_up, args=(5_000_000,))
t1.start(); t2.start(); t1.join(); t2.join()
print(f"Threaded CPU: {time.perf_counter()-start:.2f}s")   # ~same as sequential

# I/O-bound: threads ARE faster (GIL released during sleep/IO)
def io_task(): time.sleep(0.1)

start = time.perf_counter()
threads = [threading.Thread(target=io_task) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
print(f"Threaded I/O: {time.perf_counter()-start:.2f}s")   # ~0.1s not 0.4s
```

---

### Threading Lock

A `Lock` ensures only one thread accesses a shared resource at a time. Always use `with lock:` — it auto-releases even if an exception occurs.

```python
import threading

counter = 0
lock = threading.Lock()

def increment(n):
    global counter
    for _ in range(n):
        with lock:       # only one thread runs this block at a time
            counter += 1

threads = [threading.Thread(target=increment, args=(1000,)) for _ in range(5)]
for t in threads: t.start()
for t in threads: t.join()

print(counter)   # always 5000 — no race condition
```

---

### Semaphore

A `Semaphore` limits how many threads can access a resource **simultaneously**. Think of it as a bouncer at a door — only N threads allowed inside at once, others wait outside.

`with sem:` is shorthand for `sem.acquire()` + `sem.release()`. It does **not** create threads — it only controls how many can enter the block at the same time.

```
10 threads created → all reach "with sem:"
Semaphore(3) → only 3 enter → 7 wait
One finishes → next one enters → always max 3 inside
```

```python
import threading, time

sem = threading.Semaphore(3)    # at most 3 threads allowed inside at once

def use_resource(name):
    with sem:                   # thread waits here if 3 others already inside
        print(f"[{name}] working...")
        time.sleep(0.2)
        print(f"[{name}] done")
    # sem auto-released when block exits — next waiting thread enters

# Create 6 threads — but only 3 ever run simultaneously
threads = [threading.Thread(target=use_resource, args=(i,)) for i in range(6)]
for t in threads: t.start()    # launch all 6 into background
for t in threads: t.join()     # main program waits here until all 6 finish

# At most 3 "working..." lines appear at any one moment
```

> **Main program** = the default thread Python starts with. `t.start()` spawns a background thread. `t.join()` makes the main program wait until that thread finishes.

---

### `ThreadPoolExecutor`

High-level API for running tasks in a thread pool. Use `map()` for uniform tasks, `submit()` + `as_completed()` when you want results as they arrive.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def fetch(url_id):
    time.sleep(0.1)          # simulate network I/O
    return f"data-{url_id}"

# map — submit all, get results in submission order
with ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(fetch, range(8)))
    print(results[:3])   # ['data-0', 'data-1', 'data-2']

# submit + as_completed — results arrive as they finish (any order)
with ThreadPoolExecutor(max_workers=4) as ex:
    futures = {ex.submit(fetch, i): i for i in range(4)}
    for future in as_completed(futures):
        url_id = futures[future]
        print(f"url-{url_id}: {future.result()}")
```

---

### `asyncio` — Coroutines & `gather`

`asyncio` is **cooperative multitasking on a single thread**. Coroutines (`async def`) run concurrently by yielding control at `await` points. `gather` runs multiple coroutines at the same time.

```python
import asyncio, time

async def fetch(name, delay):
    print(f"[{name}] start")
    await asyncio.sleep(delay)   # non-blocking — other coroutines run here
    print(f"[{name}] done")
    return f"result-{name}"

async def main():
    start = time.perf_counter()
    # All three run concurrently — total time ~0.3s not 0.6s
    results = await asyncio.gather(
        fetch("A", 0.3),
        fetch("B", 0.2),
        fetch("C", 0.1),
    )
    print(f"Done in {time.perf_counter()-start:.2f}s")
    print(results)   # ['result-A', 'result-B', 'result-C']

asyncio.run(main())
```

---

### `asyncio.Queue` — Producer-Consumer

`asyncio.Queue` is the standard way to pass work between async producers and consumers safely.

```python
import asyncio, random

async def producer(queue):
    for i in range(5):
        item = random.randint(1, 100)
        await queue.put(item)
        print(f"Produced: {item}")
        await asyncio.sleep(0.01)
    await queue.put(None)   # sentinel: signals consumer to stop

async def consumer(queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        print(f"Consumed: {item}")
        await asyncio.sleep(0.02)
        queue.task_done()

async def main():
    queue = asyncio.Queue(maxsize=3)
    await asyncio.gather(producer(queue), consumer(queue))

asyncio.run(main())
```

---

### `asyncio.to_thread` — Blocking Code in Async

Use `asyncio.to_thread()` (Python 3.9+) to run a **synchronous blocking function** without freezing the event loop. It runs the function in a thread pool.

```python
import asyncio, time

def slow_sync_task(name):
    time.sleep(0.2)     # blocks the thread, but NOT the event loop
    return f"done-{name}"

async def main():
    start = time.perf_counter()
    # Run two blocking calls concurrently
    results = await asyncio.gather(
        asyncio.to_thread(slow_sync_task, "A"),
        asyncio.to_thread(slow_sync_task, "B"),
    )
    print(results)                                  # ['done-A', 'done-B']
    print(f"Took {time.perf_counter()-start:.2f}s") # ~0.2s not 0.4s

asyncio.run(main())
```


---

## 5. Memory & Performance

### Reference Counting

CPython's primary garbage collection method. Every object has a reference count. When it drops to 0, the object is immediately freed. `sys.getrefcount()` always returns count+1 (the call itself adds one reference).

```python
import sys

x = [1, 2, 3]
print(sys.getrefcount(x))   # 2 — x itself + the getrefcount arg

y = x                        # another reference
print(sys.getrefcount(x))   # 3

del y                        # remove one reference
print(sys.getrefcount(x))   # 2 — back to 2

# Object is freed when refcount reaches 0
```

---

### Cyclic GC and `weakref`

Reference cycles (A → B → A) prevent refcount from reaching 0. Python's cyclic garbage collector (`gc` module) finds and clears them. Use `weakref` when you want to reference an object **without preventing its deletion**.

```python
import gc, weakref

class Node:
    def __init__(self, name):
        self.name = name
        self.ref = None

a = Node("A")
b = Node("B")
a.ref = b
b.ref = a     # cycle: A → B → A

del a, b      # refcount never hits 0 due to cycle
collected = gc.collect()
print(f"Collected {collected} objects")   # 2

# weakref — doesn't prevent GC
class Cache:
    pass

obj = Cache()
weak = weakref.ref(obj)
print(weak())     # <Cache object> — still alive

del obj
print(weak())     # None — object was garbage collected
```

---

### `sys.getsizeof` and `tracemalloc`

`sys.getsizeof` gives the **shallow** size (the object itself, not what it references). `tracemalloc` tracks actual memory allocations with file and line info.

```python
import sys, tracemalloc

# getsizeof — shallow only
lst = [1, 2, 3, 4, 5]
print(sys.getsizeof(lst))         # ~120 bytes (the list object)
print(sys.getsizeof(lst[0]))      # 28 bytes (one int)

# tracemalloc — real memory usage
tracemalloc.start()
data = [i for i in range(10_000)]
snapshot = tracemalloc.take_snapshot()
top = snapshot.statistics('lineno')
print(top[0])   # shows file, line, and bytes allocated
tracemalloc.stop()
```

---

### Python Profiling

**Purpose:** Profiling identifies *where* your program spends time or memory. Without it you optimize by guessing — with it you fix the actual bottleneck.

```
Profiling answers:
  Which function is called most often?
  Which function consumes the most total time?
  Which line allocates the most memory?
  Where is the GIL blocking threads?
```

---

#### How Profiling Works Internally

Python exposes a **trace/profile hook** (`sys.setprofile`) that fires on every function call and return. Profilers attach to this hook, record timestamps and call counts, then aggregate the data.

```
Your code calls foo()
      ↓
sys.setprofile fires → profiler records: (function=foo, event=call, time=t1)
      ↓
foo() runs
      ↓
sys.setprofile fires → profiler records: (function=foo, event=return, time=t2)
      ↓
elapsed = t2 - t1  →  added to foo's cumtime
```

Two profiling modes:

| Mode | How | Cost | Accuracy |
|---|---|---|---|
| **Deterministic** (`cProfile`) | hooks every call/return | higher overhead | exact call counts |
| **Statistical** (`py-spy`, `pyinstrument`) | samples the call stack at intervals (e.g. 100Hz) | near-zero overhead | approximate counts, production-safe |

---

#### `timeit` — Benchmark a Snippet

Runs a small expression N times, eliminates noise from OS scheduling.

```python
import timeit

# Compare list vs set membership
list_time = timeit.timeit("50000 in lst", setup="lst=list(range(100000))", number=10000)
set_time  = timeit.timeit("50000 in s",   setup="s=set(range(100000))",   number=10000)
print(f"List: {list_time:.4f}s")          # slow — O(n)
print(f"Set:  {set_time:.4f}s")           # fast — O(1)
print(f"Set is {list_time/set_time:.0f}x faster!")
```

---

#### `cProfile` — Function-Level Deterministic Profiler

Built into Python stdlib. Tracks every function call — exact counts and timings.

```python
import cProfile
import pstats
import io

def slow_function():
    return sum(i * i for i in range(1_000_000))

def main():
    results = [slow_function() for _ in range(5)]
    return results

# Profile programmatically
profiler = cProfile.Profile()
profiler.enable()
main()
profiler.disable()

# Print top 10 functions by cumulative time
stream = io.StringIO()
stats = pstats.Stats(profiler, stream=stream)
stats.sort_stats("cumulative")
stats.print_stats(10)
print(stream.getvalue())

# Output columns:
#   ncalls   — how many times the function was called
#   tottime  — time spent INSIDE this function (excludes subcalls)
#   cumtime  — total time including all subcalls (usually what you care about)
#   percall  — tottime / ncalls
```

```bash
# Run from terminal — no code changes needed
python -m cProfile -s cumulative your_script.py

# Save to file for later analysis
python -m cProfile -o profile.out your_script.py
python -c "import pstats; pstats.Stats('profile.out').sort_stats('cumulative').print_stats(20)"
```

---

#### `line_profiler` — Line-by-Line Timing

`cProfile` shows which *function* is slow. `line_profiler` shows which *line* inside the function is slow.

```bash
pip install line_profiler
```

```python
# Decorate the function you want to inspect
@profile                          # injected by kernprof at runtime
def process_batch(records):
    cleaned   = [r.strip() for r in records]          # line 3
    validated = [r for r in cleaned if len(r) > 0]    # line 4
    encoded   = [r.encode("utf-8") for r in validated] # line 5
    return encoded

# Run:
# kernprof -l -v your_script.py
#
# Output:
# Line #  Hits    Time   Per Hit  % Time  Line Contents
#      3   1000  12000     12.0    55%    cleaned = ...
#      4   1000   4000      4.0    18%    validated = ...
#      5   1000   6000      6.0    27%    encoded = ...
```

---

#### `memory_profiler` — Line-by-Line Memory

```bash
pip install memory_profiler
```

```python
from memory_profiler import profile

@profile
def load_data():
    data = [i for i in range(1_000_000)]    # allocates ~35 MB
    filtered = [x for x in data if x % 2]  # another ~17 MB
    return filtered

# Run:
# python -m memory_profiler your_script.py
#
# Output:
# Line  Mem usage    Increment   Line Contents
#    3   50.0 MiB    +35.2 MiB   data = [i for i ...]
#    4   67.1 MiB    +17.1 MiB   filtered = [x for ...]
```

---

#### `tracemalloc` — Memory Snapshots (stdlib)

No install needed. Tracks which lines allocated the most memory.

```python
import tracemalloc

tracemalloc.start()

# Code under inspection
data = {i: str(i) * 100 for i in range(100_000)}

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics("lineno")

print("Top 5 memory allocations:")
for stat in top_stats[:5]:
    print(stat)

# Output:
# your_script.py:4: size=45.2 MiB, count=100000, average=474 B
```

---

#### `py-spy` — Statistical Profiler (Zero Overhead, Production-Safe)

Samples the call stack externally — no code changes, works on running processes, safe in prod.

```bash
pip install py-spy

# Profile a running process by PID
py-spy top --pid 12345

# Record a flamegraph
py-spy record -o flamegraph.svg -- python your_script.py
```

A **flamegraph** shows the call stack as nested bars — wider bar = more time spent there. The hottest path is instantly visible.

---

#### Profiling and Parallelism

| Scenario | What profiling shows | Limitation |
|---|---|---|
| **Single-threaded** | Full picture — all function times accurate | None |
| **Threading** | `cProfile` profiles only the thread it's attached to | Miss GIL contention — use `py-spy` instead |
| **Multiprocessing** | Each process needs its own profiler | Combine stats manually with `pstats.add()` |
| **asyncio** | `cProfile` misses async suspension time | Use `pyinstrument` — async-aware |
| **GIL bottleneck detection** | `py-spy --gil` shows GIL wait time per thread | Tells you when threads block each other |

```python
# Profile each process in a multiprocessing pool
import cProfile
import multiprocessing

def worker(task):
    profiler = cProfile.Profile()
    profiler.enable()
    result = heavy_computation(task)
    profiler.disable()
    profiler.dump_stats(f"worker_{multiprocessing.current_process().pid}.prof")
    return result

# Merge profiles later
import pstats
stats = pstats.Stats("worker_1234.prof")
stats.add("worker_5678.prof")
stats.sort_stats("cumulative").print_stats(10)
```

```bash
# Detect GIL contention across threads
py-spy top --pid <pid> --gil

# async-aware profiling
pip install pyinstrument
python -m pyinstrument your_async_script.py
```

---

#### Profiling Tool Cheat Sheet

| Tool | What it measures | Granularity | Overhead | Best for |
|---|---|---|---|---|
| `timeit` | snippet execution time | expression | negligible | micro-benchmarks |
| `cProfile` | function call time + count | function | medium | finding slow functions |
| `line_profiler` | time per line | line | high | pinpointing slow lines |
| `memory_profiler` | memory per line | line | high | memory bloat |
| `tracemalloc` | memory allocations | line/file | low | stdlib, no install |
| `py-spy` | CPU samples | function | near-zero | production, multi-threaded |
| `pyinstrument` | wall-clock samples | function | low | asyncio code |

---

#### Profiling Workflow

```
1. Run cProfile → find the top 3 slowest functions by cumtime
2. Run line_profiler on those functions → find the slow lines
3. Fix the hotspot (wrong data structure, redundant loop, missing cache)
4. Run timeit before/after → confirm the improvement
5. If memory is the issue: tracemalloc or memory_profiler instead
6. For production/threading: py-spy flamegraph
```

---

### `__slots__` for Memory Savings

Without `__slots__`, every instance stores its attributes in a `__dict__` (a hash map). With `__slots__`, Python uses a compact fixed array — saving memory for classes with many instances.

```python
import sys

class WithDict:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

class WithSlots:
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

a = WithDict(1, 2, 3)
b = WithSlots(1, 2, 3)

# __dict__ itself costs ~200 bytes
print(sys.getsizeof(a) + sys.getsizeof(a.__dict__))  # ~360 bytes
print(sys.getsizeof(b))                               # ~64 bytes — much less!
```

---

### `array` and `memoryview`

`array.array` stores typed numeric data much more compactly than a `list`. `memoryview` lets you slice buffers **without copying** the data.

```python
import array, sys

# list of 1000 ints — each int is a full Python object
lst = list(range(1000))
arr = array.array('i', range(1000))   # 'i' = signed int (4 bytes each)

print(sys.getsizeof(lst))   # ~8056 bytes
print(sys.getsizeof(arr))   # ~4064 bytes — ~half the size

# memoryview — zero-copy slicing
data = bytearray(b"Hello, World!")
mv = memoryview(data)
chunk = mv[7:12]              # no copy made
print(bytes(chunk))           # b'World'

mv[0] = ord('h')              # modifies the original bytearray in-place
print(data)                   # bytearray(b'hello, World!')
```

---

### `copy.copy` vs `copy.deepcopy`

`copy.copy` makes a **shallow copy** — the new object is a copy but its nested objects are still shared. `copy.deepcopy` makes a fully independent recursive copy.

```python
import copy

original = {"name": "Alice", "scores": [90, 85, 92]}

shallow = copy.copy(original)
deep    = copy.deepcopy(original)

original["scores"].append(100)

print(original["scores"])   # [90, 85, 92, 100]
print(shallow["scores"])    # [90, 85, 92, 100]  ← shares the same list!
print(deep["scores"])       # [90, 85, 92]        ← fully independent copy
```


---

## 6. Data Structures & Algorithms

### List vs Dict vs Set Complexity

Know what each operation costs so you can pick the right data structure.

```python
import timeit

setup = "d = {i: True for i in range(100000)}; lst = list(range(100000)); s = set(range(100000))"

dict_t = timeit.timeit("50000 in d",   setup=setup, number=100000)
list_t = timeit.timeit("50000 in lst", setup=setup, number=100000)
set_t  = timeit.timeit("50000 in s",   setup=setup, number=100000)

print(f"Dict 'in':  {dict_t:.4f}s — O(1)")   # fastest
print(f"Set  'in':  {set_t:.4f}s  — O(1)")   # fastest
print(f"List 'in':  {list_t:.4f}s — O(n)")   # slowest

# list.pop() vs list.pop(0)
# pop()  — O(1): removes last element
# pop(0) — O(n): shifts all elements left; use deque.popleft() instead
```

---

### `collections.deque`

A double-ended queue with **O(1) append and pop from both ends**. Use it as a queue (FIFO), stack (LIFO), or sliding window buffer.

```python
from collections import deque

# As a queue (FIFO)
q = deque()
q.append("first")
q.append("second")
q.append("third")
print(q.popleft())   # first — O(1)

# Sliding window of last N items
window = deque(maxlen=3)
for x in range(6):
    window.append(x)
    print(list(window))
# [0]
# [0, 1]
# [0, 1, 2]
# [1, 2, 3]   ← old items auto-removed
# [2, 3, 4]
# [3, 4, 5]

# BFS uses deque for O(1) popleft
def bfs(graph, start):
    visited, queue, order = {start}, deque([start]), []
    while queue:
        node = queue.popleft()
        order.append(node)
        for n in graph.get(node, []):
            if n not in visited:
                visited.add(n); queue.append(n)
    return order
```

---

### `heapq` — Priority Queue

Python's `heapq` is a **min-heap** — `heappop()` always returns the smallest item. For a max-heap, negate the values.

```python
import heapq

nums = [3, 1, 4, 1, 5, 9, 2, 6]
heapq.heapify(nums)          # convert list to heap in-place, O(n)
print(nums[0])               # 1 — smallest is always at index 0
print(heapq.heappop(nums))   # 1 — removes and returns smallest

# Max-heap trick: store negative values
max_heap = []
for n in [3, 1, 4, 1, 5, 9]:
    heapq.heappush(max_heap, -n)
print(-heapq.heappop(max_heap))   # 9 — largest

# Task priority queue: (priority, task)
tasks = []
heapq.heappush(tasks, (3, "low"))
heapq.heappush(tasks, (1, "urgent"))
heapq.heappush(tasks, (2, "normal"))
while tasks:
    p, t = heapq.heappop(tasks)
    print(f"[{p}] {t}")   # urgent, normal, low — in priority order
```

---

### `bisect` — Binary Search

`bisect` performs binary search on a **sorted list** in O(log n). Use it for fast insertions or lookups without converting to a set.

```python
import bisect

data = [1, 3, 5, 7, 9, 11]

# bisect_left: index where value would be inserted (left of duplicates)
print(bisect.bisect_left(data, 5))    # 2  — data[2] == 5

# bisect_right: index after existing value
print(bisect.bisect_right(data, 5))   # 3

# insort: insert while keeping list sorted
bisect.insort(data, 4)
print(data)   # [1, 3, 4, 5, 7, 9, 11]

# Grade lookup — classic bisect pattern
def grade(score):
    breakpoints = [60, 70, 80, 90]
    letters     = "FDCBA"
    return letters[bisect.bisect(breakpoints, score)]

for s in [45, 65, 75, 88, 95]:
    print(f"{s} -> {grade(s)}")   # F, D, C, B, A
```

---

### Floyd's Cycle Detection

Use **slow and fast pointers** to detect a cycle in O(n) time and O(1) space. After they meet, reset one pointer to head and advance both by 1 to find the cycle entry.

```python
class Node:
    def __init__(self, v): self.val = v; self.next = None

def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next           # moves 1 step
        fast = fast.next.next     # moves 2 steps
        if slow is fast: return True
    return False

def find_cycle_start(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next; fast = fast.next.next
        if slow is fast: break
    else:
        return None   # no cycle

    slow = head       # reset one pointer to head
    while slow is not fast:
        slow = slow.next; fast = fast.next   # advance both by 1
    return slow       # meeting point is cycle start

# Build: 1 -> 2 -> 3 -> 4 -> (back to 2)
n1,n2,n3,n4 = Node(1),Node(2),Node(3),Node(4)
n1.next=n2; n2.next=n3; n3.next=n4; n4.next=n2

print(has_cycle(n1))                 # True
print(find_cycle_start(n1).val)      # 2
```

---

### Binary Tree Traversals

The three DFS traversals differ only in **when** you process the current node relative to its children. BFS uses a queue to visit level by level.

```python
from collections import deque

class TreeNode:
    def __init__(self, v, l=None, r=None): self.val=v; self.left=l; self.right=r

root = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))

def inorder(n):   # left -> node -> right — gives sorted order for BST
    return inorder(n.left) + [n.val] + inorder(n.right) if n else []

def preorder(n):  # node -> left -> right — useful for copying tree
    return [n.val] + preorder(n.left) + preorder(n.right) if n else []

def postorder(n): # left -> right -> node — useful for deleting tree
    return postorder(n.left) + postorder(n.right) + [n.val] if n else []

def bfs(root):    # level by level
    q, result = deque([root]), []
    while q:
        node = q.popleft(); result.append(node.val)
        if node.left: q.append(node.left)
        if node.right: q.append(node.right)
    return result

print(inorder(root))    # [4, 2, 5, 1, 3]
print(preorder(root))   # [1, 2, 4, 5, 3]
print(bfs(root))        # [1, 2, 3, 4, 5]
```

---

### Dynamic Programming

DP solves problems by breaking them into overlapping subproblems. **Memoization** (top-down) caches recursive calls. **Tabulation** (bottom-up) builds solutions iteratively.

```python
import functools

# Top-down: @lru_cache handles caching automatically
@functools.lru_cache(maxsize=None)
def fib(n):
    if n < 2: return n
    return fib(n-1) + fib(n-2)

print(fib(50))   # 12586269025 — instant, no re-computation

# Bottom-up: O(1) space, no recursion
def fib_iter(n):
    a, b = 0, 1
    for _ in range(n): a, b = b, a+b
    return a

print(fib_iter(50))   # 12586269025

# 0/1 Knapsack — classic 2D DP
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0]*(capacity+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for w in range(capacity+1):
            dp[i][w] = dp[i-1][w]           # skip item
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w-weights[i-1]] + values[i-1])
    return dp[n][capacity]

print(knapsack([2,3,4,5],[3,4,5,6], 8))   # 10
```

---

### Two Pointers & Sliding Window

Two classic O(n) patterns that avoid nested loops.

```python
# Two pointers — find pair summing to target in sorted array
def two_sum(arr, target):
    l, r = 0, len(arr)-1
    while l < r:
        s = arr[l] + arr[r]
        if s == target: return (arr[l], arr[r])
        elif s < target: l += 1
        else: r -= 1
    return None

print(two_sum([1,2,3,4,6], 6))   # (2, 4)

# Sliding window — max sum subarray of size k
def max_window(arr, k):
    window = sum(arr[:k])
    best = window
    for i in range(k, len(arr)):
        window += arr[i] - arr[i-k]   # add new, remove old
        best = max(best, window)
    return best

print(max_window([2,1,5,1,3,2], 3))   # 9  (5+1+3)
```


---

## 7. Design Patterns

### Singleton

Ensures only **one instance** of a class ever exists. The metaclass approach is thread-safe and transparent to the caller.

```python
import threading

class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self, url="default"):
        self.url = url

db1 = Database("postgres://prod")
db2 = Database("postgres://dev")   # ignored — first instance returned

print(db1 is db2)   # True
print(db1.url)      # postgres://prod
```

---

### Factory Pattern

A **factory function** creates and returns objects based on a key. Using a registry dict is cleaner than a long `if/elif` chain and is easy to extend.

```python
from abc import ABC, abstractmethod

class Notification(ABC):
    @abstractmethod
    def send(self, msg: str) -> str: ...

class Email(Notification):
    def send(self, msg): return f"[EMAIL] {msg}"

class SMS(Notification):
    def send(self, msg): return f"[SMS] {msg}"

class Push(Notification):
    def send(self, msg): return f"[PUSH] {msg}"

# Registry: just a dict mapping key → class
_registry = {"email": Email, "sms": SMS, "push": Push}

def create_notification(channel: str) -> Notification:
    cls = _registry.get(channel.lower())
    if not cls:
        raise ValueError(f"Unknown channel: {channel}")
    return cls()

n = create_notification("email")
print(n.send("Hello!"))   # [EMAIL] Hello!
```

---

### Builder Pattern

The builder pattern uses **method chaining** (each method returns `self`) to construct complex objects step by step. A final `build()` produces the result.

```python
class QueryBuilder:
    def __init__(self):
        self._table = ""; self._cols = ["*"]
        self._where = []; self._limit = None

    def from_table(self, t):
        self._table = t; return self

    def select(self, *cols):
        self._cols = list(cols); return self

    def where(self, cond):
        self._where.append(cond); return self

    def limit(self, n):
        self._limit = n; return self

    def build(self):
        sql = f"SELECT {', '.join(self._cols)} FROM {self._table}"
        if self._where:
            sql += " WHERE " + " AND ".join(self._where)
        if self._limit:
            sql += f" LIMIT {self._limit}"
        return sql

query = (
    QueryBuilder()
    .from_table("users")
    .select("id", "name", "email")
    .where("age > 18")
    .where("active = true")
    .limit(10)
    .build()
)
print(query)
# SELECT id, name, email FROM users WHERE age > 18 AND active = true LIMIT 10
```

---

### Observer Pattern

The observer pattern lets objects **subscribe to events** and get notified automatically when something happens.

```python
from typing import Callable

class Event:
    def __init__(self):
        self._handlers: list[Callable] = []

    def __iadd__(self, fn):   # store += fn
        self._handlers.append(fn); return self

    def __isub__(self, fn):   # store -= fn
        self._handlers = [h for h in self._handlers if h != fn]; return self

    def fire(self, **kwargs):
        for h in self._handlers: h(**kwargs)

class Store:
    def __init__(self):
        self.on_purchase = Event()

    def buy(self, item, qty):
        print(f"Processing purchase: {qty}x {item}")
        self.on_purchase.fire(item=item, qty=qty)

def send_email(item, qty):  print(f"  [EMAIL] bought {qty}x {item}")
def log_analytics(item, qty): print(f"  [LOG] {item} x{qty}")

store = Store()
store.on_purchase += send_email
store.on_purchase += log_analytics

store.buy("laptop", 2)
# [EMAIL] bought 2x laptop
# [LOG] laptop x2

store.on_purchase -= send_email
store.buy("mouse", 1)   # only log_analytics fires now
```

---

### Strategy Pattern

The strategy pattern lets you **swap the algorithm** at runtime. In Python, simply pass a function — no need for a full class hierarchy.

```python
from typing import Callable

products = [
    {"name": "Laptop",  "price": 999, "rating": 4.5},
    {"name": "Mouse",   "price":  29, "rating": 4.8},
    {"name": "Monitor", "price": 399, "rating": 4.2},
]

def by_name(items):   return sorted(items, key=lambda x: x["name"])
def by_price(items):  return sorted(items, key=lambda x: x["price"])
def by_rating(items): return sorted(items, key=lambda x: -x["rating"])

def show(items, strategy: Callable = by_name):
    for p in strategy(items):
        print(f"  {p['name']:10} ${p['price']:4}  ★{p['rating']}")

print("By price:"); show(products, by_price)
# Mouse      $  29  ★4.8
# Monitor    $ 399  ★4.2
# Laptop     $ 999  ★4.5
```

---

### Context Manager

A context manager handles **setup and teardown** automatically using the `with` statement. Use a class (`__enter__`/`__exit__`) or `@contextmanager` with `yield`.

```python
from contextlib import contextmanager
import time

# Class-based
class Timer:
    def __enter__(self):
        self.start = time.perf_counter(); return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        print(f"Elapsed: {self.elapsed:.4f}s")
        return False   # False = don't suppress exceptions

with Timer() as t:
    sum(range(1_000_000))
# Elapsed: 0.0312s

# Generator-based (simpler for one-offs)
@contextmanager
def managed_transaction(name):
    print(f"BEGIN {name}")
    try:
        yield                    # your code runs here
        print(f"COMMIT {name}")
    except Exception as e:
        print(f"ROLLBACK {name}: {e}"); raise

with managed_transaction("orders"):
    print("  inserting order...")
# BEGIN orders
#   inserting order...
# COMMIT orders
```

---

### Command Pattern (Undo/Redo)

Encapsulates an operation as an object with `execute()` and `undo()`. A history stack enables undo, and a redo stack enables redo.

```python
from abc import ABC, abstractmethod
from collections import deque

class Command(ABC):
    @abstractmethod
    def execute(self): ...
    @abstractmethod
    def undo(self): ...

class TextEditor:
    def __init__(self): self.text = ""; self._history = deque(); self._redo = deque()

    def run(self, cmd: Command):
        cmd.execute(); self._history.append(cmd); self._redo.clear()

    def undo(self):
        if self._history:
            cmd = self._history.pop(); cmd.undo(); self._redo.append(cmd)

    def redo(self):
        if self._redo:
            cmd = self._redo.pop(); cmd.execute(); self._history.append(cmd)

class Insert(Command):
    def __init__(self, editor, text, pos):
        self.e, self.text, self.pos = editor, text, pos

    def execute(self):
        t = self.e.text; self.e.text = t[:self.pos] + self.text + t[self.pos:]

    def undo(self):
        t = self.e.text; self.e.text = t[:self.pos] + t[self.pos+len(self.text):]

ed = TextEditor()
ed.run(Insert(ed, "Hello", 0))
ed.run(Insert(ed, " World", 5))
print(ed.text)   # Hello World
ed.undo(); print(ed.text)   # Hello
ed.redo(); print(ed.text)   # Hello World
```


---

## 8. Testing

### `unittest` Lifecycle

The test lifecycle runs in this order: `setUpClass` → `setUp` → each `test_*` → `tearDown` → `tearDownClass`. Use `setUpClass` for expensive shared setup (e.g., DB connection). `setUp` runs fresh before every single test.

```python
import unittest

class TestMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):               # runs ONCE before all tests in this class
        print("\n[setUpClass] connecting to DB...")
        cls.multiplier = 10

    def setUp(self):                   # runs before EACH test method
        self.value = 5

    def test_multiply(self):
        self.assertEqual(self.value * self.multiplier, 50)

    def test_raises(self):
        with self.assertRaises(ZeroDivisionError):
            1 / 0

    def tearDown(self):                # runs after EACH test
        pass                           # cleanup per test

    @classmethod
    def tearDownClass(cls):            # runs ONCE after all tests
        print("[tearDownClass] closing DB...")

unittest.main(argv=[""], exit=False, verbosity=2)
```

---

### pytest — Plain Functions

pytest discovers `test_*.py` files and functions that start with `test_`. No class needed. The plain `assert` gives rich failure messages automatically.

```python
# save as test_math.py and run: pytest -v

def add(x, y): return x + y

def test_add_positive():
    assert add(2, 3) == 5          # pytest shows diff on failure

def test_add_negative():
    assert add(-1, -1) == -2

def test_raises():
    import pytest
    with pytest.raises(ZeroDivisionError):
        1 / 0

def test_float():
    import pytest
    assert 0.1 + 0.2 == pytest.approx(0.3)   # handles floating point imprecision
```

---

### Fixtures

Fixtures are **reusable setup functions**. pytest injects them by matching parameter names. Use `yield` for teardown — everything after `yield` runs as cleanup.

```python
import pytest

@pytest.fixture
def user():
    return {"id": 1, "name": "Alice", "email": "alice@example.com"}

@pytest.fixture
def temp_store():
    store = {"items": []}          # setup
    yield store                    # test runs here
    store.clear()                  # teardown — always runs

@pytest.fixture(scope="session")   # created once for entire test session
def database():
    conn = {"connected": True}
    yield conn
    print("closing DB")

def test_user_has_email(user):     # pytest injects user fixture
    assert "@" in user["email"]

def test_store_adds_item(temp_store):
    temp_store["items"].append("apple")
    assert len(temp_store["items"]) == 1
```

---

### Parametrize

Run the same test with many different inputs. pytest creates one separate test case per parameter set, shown individually in the output.

```python
import pytest

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0: return False
    return True

@pytest.mark.parametrize("n, expected", [
    (2,  True),
    (3,  True),
    (4,  False),
    (17, True),
    (1,  False),
    (0,  False),
], ids=["prime-2","prime-3","composite-4","prime-17","one","zero"])
def test_is_prime(n, expected):
    assert is_prime(n) == expected

# Output:
# PASSED test_is_prime[prime-2]
# PASSED test_is_prime[prime-3]
# PASSED test_is_prime[composite-4]
# ...
```

---

### `Mock` and `MagicMock`

`Mock` records all calls made to it. `MagicMock` also supports dunder methods (`__len__`, `__getitem__`, etc.). Use `return_value` to control what it returns, `side_effect` for sequences or exceptions.

```python
from unittest.mock import Mock, MagicMock

# Basic Mock
m = Mock()
m.fetch(1, key="val")
print(m.fetch.called)         # True
print(m.fetch.call_count)     # 1
m.fetch.assert_called_once_with(1, key="val")   # passes

# return_value
m.get.return_value = {"id": 1, "name": "Alice"}
print(m.get())                # {'id': 1, 'name': 'Alice'}

# side_effect — return a sequence, then raise an exception
m.read.side_effect = [b"chunk1", b"chunk2", EOFError("done")]
print(m.read())   # b'chunk1'
print(m.read())   # b'chunk2'
# m.read()        # raises EOFError

# MagicMock supports dunder methods
mm = MagicMock()
mm.__len__.return_value = 5
print(len(mm))    # 5
```

---

### `patch`

`patch` temporarily **replaces an object** during a test. The golden rule: patch where the name is **used**, not where it is **defined**.

```python
from unittest.mock import patch, Mock
import unittest

def get_user_name(user_id):
    import requests
    resp = requests.get(f"https://api.example.com/users/{user_id}")
    return resp.json()["name"]

class TestGetUser(unittest.TestCase):
    @patch("__main__.requests.get")   # patch where it's USED
    def test_get_user_name(self, mock_get):
        mock_get.return_value.json.return_value = {"name": "Alice"}

        result = get_user_name(1)

        mock_get.assert_called_once_with("https://api.example.com/users/1")
        self.assertEqual(result, "Alice")

# As context manager
with patch("builtins.print") as mock_print:
    print("hello")
    mock_print.assert_called_once_with("hello")
```

---

### `create_autospec`

`create_autospec` creates a mock that **enforces the real method's signature**. If you call it with wrong arguments, it raises `TypeError` immediately — catching bugs that plain `Mock` would miss.

```python
from unittest.mock import create_autospec

class PaymentGateway:
    def charge(self, amount: float, card_token: str) -> dict: ...

mock_gw = create_autospec(PaymentGateway, instance=True)
mock_gw.charge.return_value = {"success": True, "txn": "abc123"}

result = mock_gw.charge(99.99, "card_tok_123")   # correct args — works
print(result)   # {'success': True, 'txn': 'abc123'}

# mock_gw.charge(wrong_arg="bad")   # TypeError — autospec catches this!
# Plain Mock() would silently accept it and you'd never know
```


---

## 9. Advanced Internals

### Import System

When you `import foo`, Python: (1) checks `sys.modules` cache, (2) finds the module using finders in `sys.meta_path`, (3) loads and executes it, (4) caches it. All subsequent imports return the cached object.

```python
import sys, importlib

# sys.modules is the import cache
import os
print(os is sys.modules["os"])   # True — same object

# Dynamic import by string name
mod = importlib.import_module("json")
print(mod.dumps({"key": "val"}))   # '{"key": "val"}'

# Reload a module (re-executes the module file)
importlib.reload(mod)

# Inject a fake module (useful in tests)
from unittest.mock import MagicMock
sys.modules["fake_lib"] = MagicMock()
import fake_lib                  # won't fail — uses the mock
print(fake_lib.anything())       # MagicMock()
```

---

### Bytecode and `dis`

Python source is compiled to **bytecode** (a series of low-level instructions) before execution. Use `dis.dis()` to inspect what the interpreter actually runs.

```python
import dis

def add(x, y):
    return x + y

dis.dis(add)
# Disassembly:
#   2           0 LOAD_FAST    0 (x)
#               2 LOAD_FAST    1 (y)
#               4 BINARY_OP   0 (+)
#               6 RETURN_VALUE

# Python caches compiled bytecode in __pycache__/ as .pyc files
# Bytecode is version-specific — Python 3.11 and 3.12 have different opcodes
```

---

### Frame Objects

Every function call creates a **frame object** that stores local variables, the code object, and execution state. Useful for debugging, tracing, and advanced introspection.

```python
import sys

def inspect_frame():
    frame = sys._getframe(0)     # current frame
    print(f"Function: {frame.f_code.co_name}")    # inspect_frame
    print(f"File:     {frame.f_code.co_filename}")
    print(f"Line:     {frame.f_lineno}")
    print(f"Locals:   {list(frame.f_locals.keys())}")

def outer():
    x = 10
    def inner():
        frame = sys._getframe(1)   # caller's frame
        print(f"Caller locals: {frame.f_locals}")
    inner()

inspect_frame()
outer()
```

---

### `exec` and `eval`

`eval` evaluates a **single expression** and returns its value. `exec` executes **statements** (no return value). Always restrict `globals` when running untrusted code.

```python
# eval — expression only
result = eval("2 ** 10 + len('hello')")
print(result)   # 1029

# exec — statements
code = """
def greet(name):
    return f"Hello, {name}!"
"""
namespace = {}
exec(code, namespace)
print(namespace["greet"]("Alice"))   # Hello, Alice!

# Sandboxed eval — restrict builtins
safe_result = eval("1 + 2 * 3", {"__builtins__": {}})
print(safe_result)   # 7
# eval("__import__('os')", {"__builtins__": {}})   # NameError — blocked
```

---

### `ast` Module

The `ast` module parses Python source into an **Abstract Syntax Tree** you can inspect and transform. Used by linters (`pylint`, `flake8`), type checkers (`mypy`), and code generators.

```python
import ast

source = """
x = 1 + 2
print(x * 3)
"""

tree = ast.parse(source)
print(ast.dump(tree, indent=2))   # shows the full tree structure

# Walk the tree to find all function calls
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            print(f"Call to: {node.func.id}")   # print

# Compile and run modified AST
code = compile(tree, "<string>", "exec")
exec(code)   # 9
```

---

### `weakref` — Prevent Memory Leaks in Caches

A `weakref` holds a reference to an object **without preventing** garbage collection. When the object is GC'd, the weakref returns `None`. Essential for caches and event systems.

```python
import weakref

class Image:
    def __init__(self, name): self.name = name
    def __del__(self): print(f"[GC] {self.name} deleted")

# WeakValueDictionary: entries auto-removed when values are GC'd
cache = weakref.WeakValueDictionary()

img = Image("logo.png")
cache["logo"] = img
print(cache["logo"].name)   # logo.png

del img                     # remove the only strong reference
import gc; gc.collect()
print(list(cache.keys()))   # [] — entry auto-removed!
```

---

### `contextvars` — Per-Coroutine Storage

`contextvars.ContextVar` provides storage that's **isolated per async task** (like `threading.local` but for coroutines). Each `asyncio.create_task` automatically gets its own copy of the context.

```python
import asyncio
from contextvars import ContextVar

request_id = ContextVar("request_id", default="none")

async def handle_request(rid):
    token = request_id.set(rid)          # set in this coroutine's context
    await asyncio.sleep(0.01)            # other coroutines run here
    print(f"Request {request_id.get()}") # sees its own value, not others'
    request_id.reset(token)              # restore previous value

async def main():
    await asyncio.gather(
        handle_request("req-1"),
        handle_request("req-2"),
        handle_request("req-3"),
    )
    # Each prints its own ID — no cross-contamination

asyncio.run(main())
# Request req-1
# Request req-2
# Request req-3
```

---

### `__init_subclass__`

A hook called automatically when a class is **subclassed**. Cleaner than a metaclass for simple plugin registration or validation — no metaclass boilerplate needed.

```python
class Plugin:
    _registry = {}

    def __init_subclass__(cls, plugin_name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        name = plugin_name or cls.__name__.lower()
        Plugin._registry[name] = cls
        print(f"Registered plugin: {name}")

class CSVReader(Plugin, plugin_name="csv"):
    def read(self, path): return f"reading CSV: {path}"

class JSONReader(Plugin, plugin_name="json"):
    def read(self, path): return f"reading JSON: {path}"

# Registered plugin: csv
# Registered plugin: json

reader_cls = Plugin._registry["csv"]
print(reader_cls().read("data.csv"))   # reading CSV: data.csv
```


---

## 10. Type System & Protocols

### Type Hints Basics

Type hints are **optional metadata** — Python doesn't enforce them at runtime. They exist for tools like `mypy` and IDEs. Use `X | Y` (Python 3.10+) or `Union[X, Y]` for multiple types.

```python
from typing import Optional, Union

def greet(name: str, times: int = 1) -> str:
    return (f"Hello, {name}! " * times).strip()

# Optional — value can be None
def find_user(uid: int) -> Optional[str]:
    db = {1: "Alice", 2: "Bob"}
    return db.get(uid)   # returns str or None

# Union / | syntax (Python 3.10+)
def process(value: int | str) -> str:
    if isinstance(value, int): return str(value * 2)
    return value.upper()

# Type hints are NOT enforced at runtime!
print(process(5))         # '10'
print(process("hello"))   # 'HELLO'
print(process([1, 2]))    # no error at runtime — but mypy would flag it
```

---

### `TypeVar` and `Generic[T]`

`TypeVar` creates a placeholder type. `Generic[T]` lets you write classes that work with any type while preserving type information across methods.

```python
from typing import TypeVar, Generic

T = TypeVar("T")

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        if not self._items: raise IndexError("empty stack")
        return self._items.pop()

    def peek(self) -> T:
        return self._items[-1]

s: Stack[int] = Stack()
s.push(1); s.push(2); s.push(3)
print(s.pop())    # 3
print(s.peek())   # 2

# mypy knows: s.push("hello") would be a type error
# At runtime, generics are ERASED — Stack[int] == Stack[str] as a class
```

---

### `Protocol` — Structural Typing

`Protocol` defines an **interface by behaviour** (duck typing + static types). Any class with the right methods satisfies the protocol — no need to inherit from it.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> str: ...

class Circle:
    def draw(self) -> str: return "Drawing Circle"

class Square:
    def draw(self) -> str: return "Drawing Square"

class Triangle:
    def draw(self) -> str: return "Drawing Triangle"

def render(shape: Drawable) -> None:
    print(shape.draw())

# All three work — none inherit from Drawable!
for shape in [Circle(), Square(), Triangle()]:
    render(shape)

# runtime_checkable enables isinstance
print(isinstance(Circle(), Drawable))   # True
print(isinstance("hello", Drawable))   # False — no draw() method
```

---

### `TypedDict`

`TypedDict` gives **type hints for dictionaries** with specific keys. It's just a `dict` at runtime — but `mypy` checks key names and value types.

```python
from typing import TypedDict, NotRequired

class User(TypedDict):
    id: int
    name: str
    email: str

class Employee(TypedDict):
    id: int
    name: str
    department: NotRequired[str]   # this key is optional

def greet_user(u: User) -> str:
    return f"Hello, {u['name']}!"

user: User = {"id": 1, "name": "Alice", "email": "alice@example.com"}
print(greet_user(user))   # Hello, Alice!

emp: Employee = {"id": 2, "name": "Bob"}   # department is optional — OK
print(emp)   # {'id': 2, 'name': 'Bob'}
```

---

### `Annotated` — Metadata in Type Hints

`Annotated[T, metadata]` attaches extra information to a type. Libraries like Pydantic and FastAPI use this for validation rules without requiring special subclasses.

```python
from typing import Annotated, get_args

class Gt:
    def __init__(self, v): self.v = v

class MaxLen:
    def __init__(self, n): self.n = n

Age      = Annotated[int, Gt(0)]
Username = Annotated[str, MaxLen(20)]

def validate(value, annotation):
    base, *validators = get_args(annotation)
    if not isinstance(value, base):
        raise TypeError(f"Expected {base.__name__}")
    for v in validators:
        if isinstance(v, Gt) and value <= v.v:
            raise ValueError(f"Must be > {v.v}")
    return value

print(validate(25, Age))    # 25
try:
    validate(-5, Age)       # ValueError: Must be > 0
except ValueError as e:
    print(e)
```

---

### `@overload`

`@overload` lets you declare **different return types** depending on the input type, for type checkers. Only the real implementation actually runs — the `@overload` stubs are ignored at runtime.

```python
from typing import overload

@overload
def process(x: int) -> str: ...    # stub — only for mypy

@overload
def process(x: str) -> int: ...    # stub — only for mypy

def process(x):                    # real implementation
    if isinstance(x, int): return str(x * 2)
    if isinstance(x, str): return len(x)
    raise TypeError

print(process(5))       # '10'    — mypy infers return type is str
print(process("hi"))    # 2       — mypy infers return type is int
```

---

### Pydantic v2 — Runtime Validation

Pydantic `BaseModel` validates data **at runtime** when the object is created. Fields use standard type hints. Pydantic v2 is ~5-50x faster than v1 (Rust-based core).

```python
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Annotated

class User(BaseModel):
    id: int
    name: str = Field(min_length=1, max_length=50)
    age: Annotated[int, Field(gt=0, lt=150)]
    email: str

    @field_validator("email")
    @classmethod
    def valid_email(cls, v: str) -> str:
        if "@" not in v: raise ValueError("Invalid email")
        return v.lower()

u = User(id=1, name="Alice", age=30, email="ALICE@EXAMPLE.COM")
print(u.email)          # alice@example.com  — normalized
print(u.model_dump())   # {'id': 1, 'name': 'Alice', 'age': 30, 'email': '...'}

try:
    User(id=1, name="", age=-5, email="bad")
except ValidationError as e:
    for err in e.errors():
        print(f"{err['loc']}: {err['msg']}")
# ('name',): String should have at least 1 character
# ('age',):  Input should be greater than 0
# ('email'): Invalid email
```

---

### `NewType` and `cast`

`NewType` creates a **distinct type alias** that prevents accidentally mixing IDs. `cast` tells mypy "trust me, this is type T" — it's a no-op at runtime.

```python
from typing import NewType, cast, Any

# NewType — distinct types, prevents mixing
UserId    = NewType("UserId", int)
ProductId = NewType("ProductId", int)

def get_user(uid: UserId) -> str: return f"User {uid}"

uid = UserId(42)
pid = ProductId(42)
print(get_user(uid))     # User 42
# get_user(pid)          # mypy ERROR: ProductId is not UserId
print(type(uid))         # <class 'int'> — no-op at runtime

# cast — lie to mypy (use sparingly, prefer isinstance narrowing)
raw: Any = {"name": "Alice", "age": 30}
typed = cast(dict[str, str], raw)   # mypy trusts this
print(typed["name"])    # Alice — works at runtime
# typed["age"] + " years"  # runtime error — it's actually int, not str
```

---

### `TYPE_CHECKING` — Avoid Circular Imports

`TYPE_CHECKING` is `False` at runtime but `True` when `mypy` runs. Use it to import types that would cause circular imports or slow down startup.

```python
from __future__ import annotations   # enables string annotations (lazy evaluation)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections import OrderedDict   # only imported during type checking

def process(data: "OrderedDict") -> None:   # string annotation — not evaluated at runtime
    pass

# At runtime: OrderedDict is never imported — no circular import risk
# mypy sees: the type annotation is OrderedDict — full type checking works
```

---

## Quick Decision Guide

```
Concurrency:
  Many I/O tasks      → asyncio + gather/create_task
  Simple I/O          → ThreadPoolExecutor
  CPU-bound           → ProcessPoolExecutor
  Blocking in async   → asyncio.to_thread()

Data Structure:
  Queue (FIFO)        → collections.deque
  Priority queue      → heapq (min-heap; negate for max)
  Sorted search       → bisect on sorted list
  Membership test     → set  (O(1))
  Count items         → collections.Counter
  Default values      → collections.defaultdict

OOP Tool:
  Enforce interface   → ABC + @abstractmethod
  Duck typing + types → Protocol
  Computed attribute  → @property
  Reuse behavior      → Mixin
  Control class creation → Metaclass / __init_subclass__

Type Hint:
  Nullable value      → T | None  (or Optional[T])
  Multiple types      → A | B  (or Union[A, B])
  Exact values        → Literal["a", "b"]
  Generic class       → Generic[T]
  Duck typing         → Protocol
  Runtime validation  → Pydantic BaseModel

Testing:
  Simple assertions   → pytest plain functions
  Shared setup        → @pytest.fixture(scope="session")
  Multiple inputs     → @pytest.mark.parametrize
  Replace external    → unittest.mock.patch
  Enforce signatures  → create_autospec
```

---

## 11. Processing Huge Data on Limited Memory

When you have large datasets but constrained RAM, the strategy is: **never load everything at once — stream, chunk, and parallelize IO without wasting CPU threads on waiting.**

---

### The Core Problem

```
Naive approach:
  data = load_entire_file()    # 10 GB file → OOM crash
  results = process(data)

Correct approach:
  for chunk in stream_file():  # loads 64 MB at a time
      results = process(chunk) # process → emit → discard
```

Think of your Python program as a factory. It has three possible bottlenecks — and each one has a different fix:

```
                        Your Python Program
                               │
              ┌────────────────┼─────────────────┐
              │                │                 │
           MEMORY             CPU               IO
        "storage room"    "workers"         "delivery trucks"
              │                │                 │
       Can only hold      Workers are       Trucks are slow
       10 boxes at once   always busy       (disk, network)
       but you have       (processing       Program waits
       10,000 boxes       data non-stop)    doing nothing
              │                │                 │
           SOLUTION         SOLUTION          SOLUTION
       Don't load all    Add more workers   While one truck
       10,000 boxes.     (processes) that   is out, send
       Bring in 10,      run in parallel    more trucks
       process them,     on multiple        (async / threads)
       discard, repeat.  CPU cores.
```

**Axis 1 — Memory ("the storage room")**

Your RAM is a storage room. If you load a 10 GB file, you're trying to stack 10 GB of boxes in a room that only holds 2 GB — it crashes. The fix: bring in a small batch, process it, throw it away, bring in the next batch. This is what generators and chunked reads do. **The room never overflows.**

**Axis 2 — CPU ("the workers")**

Your CPU cores are workers. By default Python uses **only one worker at a time (due to the GIL). If each chunk needs heavy computation (math, parsing, ML inference), one worker is a bottleneck**. The fix: hire more workers — `ProcessPoolExecutor` spawns multiple Python processes, each on its own CPU core. **All cores work in parallel.**

**Axis 3 — IO ("the delivery trucks")**

IO means waiting — reading from disk, calling an API, querying a database. While your program waits for a response, the CPU sits completely idle, like a worker waiting for a truck to return before sending the next one. The fix: send all the trucks at once, switch to other tasks while waiting. `asyncio` and `ThreadPoolExecutor` both do this — your program does useful work while IO is in-flight. **No idle waiting.**

---

**Which bottleneck do you have?** Profile first, then pick the fix:

| Symptom | Bottleneck | Fix |
|---|---|---|
| Program crashes with `MemoryError` | Memory | Generators + chunked reads |
| CPU is at 100%, program is slow | CPU | `ProcessPoolExecutor` |
| CPU is near 0%, program is slow | IO | `asyncio` or `ThreadPoolExecutor` |
| All three | All three | Stream chunks → parallel process → async IO |

---

### Strategy 1 — Generators for Zero-Copy Streaming

Never materialize the full dataset. Use generators to produce one record at a time.

```python
def read_large_csv(path, chunk_size=10_000):
    """Yields one chunk (list of rows) at a time — constant memory."""
    import csv
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        chunk = []
        for row in reader:
            chunk.append(row)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []          # discard processed chunk → GC frees it
        if chunk:
            yield chunk

def transform(row: dict) -> dict:
    return {"id": row["id"], "value": float(row["value"]) * 1.1}

# Only chunk_size rows ever live in memory at once
for chunk in read_large_csv("huge.csv"):
    processed = [transform(r) for r in chunk]
    write_output(processed)
```

**Memory cost:** `O(chunk_size)` regardless of file size.

---

### Strategy 2 — Chunked Processing + `ProcessPoolExecutor` (CPU-bound)

If the transformation is CPU-heavy (parsing, math, ML inference), use multiple processes to bypass the GIL.

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_chunk(chunk: list) -> list:
    """CPU-heavy work — runs in a separate process, no GIL."""
    return [transform(row) for row in chunk]

def chunked_parallel(path, chunk_size=50_000, max_workers=4):
    """
    max_workers = number of CPU cores to use.
    Keep this <= os.cpu_count() - 1 to leave one core for the OS.
    Each worker holds only one chunk in its subprocess memory.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for i, chunk in enumerate(read_large_csv(path, chunk_size)):
            # Submit chunk to pool — chunk is serialized (pickled) to subprocess
            future = pool.submit(process_chunk, chunk)
            futures[future] = i

        for future in as_completed(futures):
            result = future.result()
            write_output(result)           # write as results arrive
            # future + result go out of scope → GC frees them

chunked_parallel("huge.csv", chunk_size=50_000, max_workers=4)
```

**Memory model:**
```
Main process:       one chunk in flight (being submitted)
Worker process 1:   one chunk being processed
Worker process 2:   one chunk being processed
Worker process N:   one chunk being processed
─────────────────────────────────────────────────
Total RAM ≈  (max_workers + 1) × chunk_size × row_size
```

Tune `chunk_size` to fit: `chunk_size ≈ available_RAM / (max_workers + 2)`.

---

### Strategy 3 — `ThreadPoolExecutor` (IO-bound, not CPU-bound)

If the bottleneck is **IO** (reading from S3, calling an API, writing to DB) and the processing itself is light, threads are sufficient — they release the GIL during IO.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

def fetch_and_process(record: dict) -> dict:
    """IO-bound: HTTP call releases GIL → other threads run during wait."""
    resp = requests.get(f"https://api.example.com/enrich/{record['id']}")
    record["enriched"] = resp.json()
    return record

def parallel_io(records, max_workers=20):
    """
    max_workers can be much higher than CPU count for IO-bound work.
    20-50 threads is common for HTTP calls — most are just waiting.
    """
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(fetch_and_process, r): r for r in records}
        for future in as_completed(futures):
            yield future.result()          # stream results out as they arrive
```

**When to use threads vs processes:**

| Work type | Tool | Why |
|---|---|---|
| CPU-heavy (math, parsing) | `ProcessPoolExecutor` | Bypasses GIL, true parallelism |
| IO-heavy (HTTP, DB, disk) | `ThreadPoolExecutor` | GIL released during IO, low overhead |
| Mixed IO + light CPU | `ThreadPoolExecutor` | Simpler, sufficient |

---

### Strategy 4 — `asyncio` for Massive IO Concurrency

`asyncio` uses a **single thread** with an event loop. When a coroutine awaits IO, the event loop switches to another coroutine — no thread switching overhead, no GIL contention.

```
Threads (10 workers):                asyncio (single thread):
  Thread 1: [work]...[wait]           Coroutine 1: [work] → await → suspended
  Thread 2: [work]...[wait]           Coroutine 2: [work] → await → suspended
  Thread 3: [work]...[wait]     →     Coroutine 3: [work] → await → suspended
  ...                                 ...
  OS schedules all 10 threads         Event loop schedules thousands of coroutines
  Context switch cost × 10            Context switch cost ≈ 0
```

```python
import asyncio
import aiohttp
import aiofiles

async def fetch(session: aiohttp.ClientSession, record: dict) -> dict:
    async with session.get(f"https://api.example.com/{record['id']}") as resp:
        record["data"] = await resp.json()
    return record

async def stream_and_enrich(path: str):
    """
    Reads file line-by-line (streaming, low memory).
    Fires thousands of HTTP requests concurrently (asyncio, no threads).
    """
    semaphore = asyncio.Semaphore(100)   # cap at 100 in-flight requests

    async def bounded_fetch(session, record):
        async with semaphore:            # backpressure — prevent memory explosion
            return await fetch(session, record)

    async with aiohttp.ClientSession() as session:
        async with aiofiles.open(path) as f:
            tasks = []
            async for line in f:                     # streams file, no full load
                record = parse_line(line)
                tasks.append(bounded_fetch(session, record))

                if len(tasks) == 500:                # batch to limit task queue size
                    results = await asyncio.gather(*tasks)
                    write_output(results)
                    tasks = []

            if tasks:
                results = await asyncio.gather(*tasks)
                write_output(results)

asyncio.run(stream_and_enrich("huge.csv"))
```

**Why `Semaphore` is critical for memory:** without it, `gather` creates all tasks upfront → all responses land in memory simultaneously → OOM. Semaphore = backpressure valve.

---

### How asyncio Helps with Parallelism

`asyncio` is **concurrent but not parallel** — one thread, many interleaved tasks.

```
                    Event Loop (single thread)
                           │
         ┌─────────────────┼──────────────────┐
         │                 │                  │
   Coro A: fetch()   Coro B: fetch()    Coro C: fetch()
         │                 │                  │
      await IO          await IO           await IO
         │                 │                  │
   [suspended]        [running]          [suspended]
                           │
                      IO completes
                           │
                   Coro B: process result → done
                           │
                   Coro A: resumes...
```

Where asyncio **does** give parallelism:

| Scenario | asyncio behaviour |
|---|---|
| 1000 HTTP requests | All in-flight simultaneously, one thread |
| Reading 50 files from S3 | All reads overlap, no thread pool needed |
| DB queries to Postgres (asyncpg) | Queries overlap while DB processes |
| Writing results while fetching | Fetch + write interleave on same thread |

Where asyncio **does not** give parallelism:

| Scenario | Problem | Fix |
|---|---|---|
| CPU-heavy transformation | Blocks the event loop | Offload to `ProcessPoolExecutor` |
| Calling a blocking library (requests, psycopg2) | Blocks event loop | Use `asyncio.to_thread()` |

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

pool = ProcessPoolExecutor(max_workers=4)

async def pipeline(records):
    """
    asyncio handles IO concurrency.
    ProcessPoolExecutor handles CPU parallelism.
    Both run together without blocking each other.
    """
    async with aiohttp.ClientSession() as session:
        for record in records:
            # IO: non-blocking, runs on event loop
            raw = await fetch(session, record)

            # CPU: offloaded to process pool, event loop stays free
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(pool, heavy_transform, raw)

            yield result
```

---

### Strategy 5 — `mmap` for Random Access Without Loading

For files needing random access (not sequential), `mmap` maps the file into virtual memory — the OS pages in only what you touch.

```python
import mmap

with open("huge_binary.dat", "rb") as f:
    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        # Access any byte range without loading the full file
        header = mm[0:128]
        record = mm[1_000_000:1_001_024]   # OS fetches only this page
```

---

### Profiling to Find the Real Bottleneck First

Before choosing a strategy, profile to find where time actually goes.

```python
import cProfile, pstats, io

profiler = cProfile.Profile()
profiler.enable()
process_sample_chunk(sample)      # run on small sample first
profiler.disable()

s = io.StringIO()
pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats(10)
print(s.getvalue())

# If tottime is high in your transform() → CPU-bound → ProcessPoolExecutor
# If tottime is low but cumtime is high in read/write → IO-bound → asyncio / threads
```

```bash
# Visual flamegraph — fastest way to see the bottleneck
pip install py-spy
py-spy record -o flamegraph.svg -- python process_data.py
```

---

### Decision Tree

```
Start: large data, limited memory
           │
           ▼
  Are you loading all data at once?
  YES → switch to generators / chunked reads first
           │
           ▼
  Profile: where is time spent?
           │
    ┌──────┴──────┐
  IO-bound      CPU-bound
    │               │
  asyncio      ProcessPoolExecutor
  or threads    (max_workers = CPU cores - 1)
    │               │
  Semaphore     chunk_size = RAM / (workers + 2)
  for backpressure
```

---

### Quick Reference

| Scenario | Tool | Key config |
|---|---|---|
| Sequential file too big for RAM | Generator + chunked read | `chunk_size` controls memory |
| CPU-heavy transform on chunks | `ProcessPoolExecutor` | `max_workers ≤ cpu_count - 1` |
| Thousands of HTTP/DB calls | `asyncio` + `aiohttp` | `Semaphore(N)` for backpressure |
| Blocking lib inside async | `asyncio.to_thread()` | Wraps sync call without blocking loop |
| CPU + IO mixed pipeline | `asyncio` + `run_in_executor` | IO on event loop, CPU on process pool |
| Random access to huge file | `mmap` | OS pages only touched regions |
| Find the actual bottleneck | `cProfile` → `py-spy` flamegraph | Profile before optimizing |