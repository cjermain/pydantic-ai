# Remaining Issues in Per-Minute Rate Limiting Prototype

## CRITICAL ISSUES

### 1. ⚠️ BREAKING CHANGE: RunContext Requires usage_history

**Severity:** CRITICAL - Breaks all existing code
**Location:** `pydantic_ai_slim/pydantic_ai/_run_context.py:38`

**Issue:**
We added `usage_history: UsageHistory` as a **required** field to `RunContext`, but this breaks ALL existing code that creates `RunContext` objects.

**Evidence:**
```python
# From test_toolsets.py:37
def build_run_context(deps: T, run_step: int = 0) -> RunContext[T]:
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),  # ✅ Has this
        prompt=None,
        messages=[],
        run_step=run_step,
        # ❌ Missing usage_history - BREAKS!
    )
```

**Test Failures:**
```
tests/test_toolsets.py::test_function_toolset FAILED
TypeError: RunContext.__init__() missing 1 required keyword-only argument: 'usage_history'
```

**Impact:**
- ALL tests that create RunContext manually will fail
- ALL user code that creates RunContext will break
- This is a major breaking change to the public API

**Fix Required:**
Make `usage_history` optional with a default value:
```python
@dataclasses.dataclass(repr=False, kw_only=True)
class RunContext(Generic[RunContextAgentDepsT]):
    deps: RunContextAgentDepsT
    model: Model
    usage: RunUsage
    usage_history: UsageHistory = field(default_factory=lambda: __import__('pydantic_ai.usage', fromlist=['UsageHistory']).UsageHistory())
    # OR
    usage_history: UsageHistory | None = None  # Then handle None everywhere
```

---

### 2. ⚠️ RACE CONDITION STILL EXISTS: add_entry() Not Thread-Safe

**Severity:** CRITICAL - Data corruption possible
**Location:** `pydantic_ai_slim/pydantic_ai/_agent_graph.py:532`

**Issue:**
`add_entry()` is called **WITHOUT holding the lock**, allowing race conditions.

**Current Code:**
```python
# _agent_graph.py:530-532
# Add response to usage history for per-minute rate limiting
if ctx.deps.usage_limits.has_per_minute_limits():
    ctx.state.usage_history.add_entry(response.usage)  # ❌ NO LOCK!
```

**The Problem:**
1. Lock is only used in `check_per_minute_limits_*()` methods
2. `add_entry()` itself is NOT protected
3. Multiple concurrent responses can call `add_entry()` simultaneously
4. List modifications are NOT atomic in Python

**Race Condition Scenario:**
```
Time    Thread 1                    Thread 2
----    --------                    --------
T1      add_entry() starts
T2      entries.append(...)         add_entry() starts
T3                                  entries.append(...)
T4      Both modifications happen simultaneously → DATA CORRUPTION
```

**Fix Required:**
Either:
1. Make `add_entry()` async and acquire lock
2. OR wrap the call in the lock
3. OR use a thread-safe queue

**Recommended Fix:**
```python
# In usage.py
async def add_entry_atomic(self, usage: RequestUsage, timestamp: float | None = None) -> None:
    """Thread-safe add entry."""
    if timestamp is None:
        timestamp = time.monotonic()
    async with self._lock:
        self.entries.append(TimedUsageEntry(timestamp=timestamp, usage=usage))

# In _agent_graph.py
if ctx.deps.usage_limits.has_per_minute_limits():
    await ctx.state.usage_history.add_entry_atomic(response.usage)
```

---

### 3. ⚠️ RACE CONDITION: cleanup_old_entries() Not Thread-Safe

**Severity:** CRITICAL - Data corruption possible
**Location:** `pydantic_ai_slim/pydantic_ai/_agent_graph.py:500`

**Issue:**
`cleanup_old_entries()` modifies the list **WITHOUT holding the lock**.

**Current Code:**
```python
# _agent_graph.py:498-500
# Always clean up old usage history entries to prevent memory growth
# (even if per-minute limits are disabled now, they may have been enabled earlier)
ctx.state.usage_history.cleanup_old_entries(60.0)  # ❌ NO LOCK!
```

**The Problem:**
1. List reassignment `self.entries = [...]` is NOT atomic
2. If another thread reads during cleanup, it could see partial state
3. List comprehension creates new list, then assigns - race window exists

**Fix Required:**
```python
# In usage.py
async def cleanup_old_entries_atomic(self, window_seconds: float) -> None:
    """Thread-safe cleanup."""
    cutoff = time.monotonic() - window_seconds
    async with self._lock:
        self.entries = [e for e in self.entries if e.timestamp >= cutoff]

# In _agent_graph.py
await ctx.state.usage_history.cleanup_old_entries_atomic(60.0)
```

---

### 4. ⚠️ get_usage_in_window() Not Thread-Safe

**Severity:** HIGH - Incorrect calculations possible
**Location:** `pydantic_ai_slim/pydantic_ai/usage.py:300`

**Issue:**
`get_usage_in_window()` reads the list WITHOUT lock protection.

**Current Code:**
```python
def get_usage_in_window(self, window_seconds: float) -> RunUsage:
    cutoff = time.monotonic() - window_seconds
    total = RunUsage()
    requests = 0
    for entry in self.entries:  # ❌ Reads without lock!
        if entry.timestamp >= cutoff:
            total.incr(entry.usage)
            requests += 1
    return total
```

**The Problem:**
1. While iterating, another thread could modify `self.entries`
2. Could raise `RuntimeError: list changed size during iteration`
3. Could return incorrect calculations if entries are added/removed mid-iteration

**Note:**
Currently this is called within the lock in `check_per_minute_limits_*()` methods, BUT:
- It's a public method that could be called from anywhere
- No documentation warns about this requirement
- Very fragile design

**Fix Required:**
Document that it must be called within lock context, OR make it acquire the lock itself.

---

## HIGH SEVERITY ISSUES

### 5. 🔴 Temporal/Durable Execution: TemporalRunContext Broken

**Severity:** HIGH - Breaks Temporal integration
**Location:** `pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_run_context.py`

**Issue:**
`TemporalRunContext` extends `RunContext` but doesn't provide `usage_history`.

**Current Code (probably):**
```python
class TemporalRunContext(RunContext[AgentDepsT]):
    # Doesn't override __init__ to provide usage_history
    # Will inherit the requirement from parent
```

**Impact:**
- Temporal workflows will fail to create RunContext
- Breaks durable execution entirely

**Fix:** Update TemporalRunContext to provide usage_history

---

### 6. 🔴 Tests: Multiple Test Files Will Fail

**Severity:** HIGH - CI/CD will be red
**Locations:** Multiple test files

**Affected Files:**
- `tests/test_toolsets.py` (confirmed failing)
- `tests/test_fastmcp.py` (likely)
- `tests/test_mcp.py` (likely)
- Any test that manually creates RunContext

**Fix:** Update all test helper functions to provide usage_history

---

## MEDIUM SEVERITY ISSUES

### 7. 🟡 Lock Cannot Be Serialized/Copied Properly

**Severity:** MEDIUM - Breaks state persistence
**Location:** `pydantic_ai_slim/pydantic_ai/usage.py:282`

**Issue:**
`asyncio.Lock` in a dataclass causes issues with:
- Deep copying (used in many places)
- Multiprocessing
- State persistence

**Current Code:**
```python
@dataclass
class UsageHistory:
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
```

**The Problem:**
```python
from copy import deepcopy
history = UsageHistory()
history2 = deepcopy(history)  # Both share same lock! Not independent.
```

**Test:**
```python
>>> from copy import deepcopy
>>> import asyncio
>>> lock1 = asyncio.Lock()
>>> lock2 = deepcopy(lock1)
>>> lock1 is lock2
True  # ❌ Same object! Not a true copy!
```

**Impact:**
- Deep copying doesn't create independent locks
- Multi-agent scenarios might share locks unintentionally
- Multiprocessing will fail (locks can't cross process boundaries)

**Fix:** Need custom `__deepcopy__` method

---

### 8. 🟡 No Type Hints for Async Methods

**Severity:** MEDIUM - Type safety issue
**Location:** Multiple

**Issue:**
`check_per_minute_limits_*()` methods are now async but return `None`, not `Awaitable[None]`.

**Impact:**
- Type checkers might complain
- IDE autocomplete might be confused

**Fix:** Ensure proper async type hints

---

## LOW SEVERITY / MINOR ISSUES

### 9. 🟢 Inconsistent Method Naming

**Severity:** LOW
**Issue:** We have both sync and async versions but no clear naming pattern

**Current:**
- `add_entry()` - sync (NOT thread-safe!)
- `check_per_minute_limits_before_request()` - async (thread-safe)

**Recommendation:**
- `add_entry()` - sync, not thread-safe
- `add_entry_atomic()` - async, thread-safe
- `cleanup_old_entries()` - sync, not thread-safe
- `cleanup_old_entries_atomic()` - async, thread-safe

---

### 10. 🟢 Missing __all__ Export

**Severity:** LOW
**Issue:** `UsageHistory` and `TimedUsageEntry` are in `__all__` but are they public API?

**Question:** Should these be exported or kept internal?

---

## DOCUMENTATION ISSUES

### 11. 📄 No Documentation on Thread Safety Requirements

**Severity:** MEDIUM

**Issue:**
Users don't know:
- Which methods are thread-safe
- That `get_usage_in_window()` must be called within lock
- That `add_entry()` is NOT thread-safe

**Fix:** Add clear docstrings documenting thread safety

---

### 12. 📄 No Migration Guide

**Severity:** HIGH

**Issue:**
This is a BREAKING CHANGE but there's no migration guide for users.

**Needed:**
- How to update code that creates RunContext
- How to pass usage_history between agents
- Examples

---

## TESTING GAPS

### 13. 🧪 No Concurrent Request Tests

**Severity:** HIGH

**Issue:**
We have NO tests that verify thread safety with actual concurrent requests.

**Needed:**
- Test with multiple concurrent tool calls
- Test with multiple concurrent streaming responses
- Test that limits are enforced correctly under concurrency

---

### 14. 🧪 No Multi-Agent Integration Tests

**Severity:** MEDIUM

**Issue:**
No tests verify that `usage_history` is properly passed between agents.

**Needed:**
- Test controller → delegate agent with usage_history
- Test that rate limits work across agents
- Test that usage_history is shared correctly

---

## SUMMARY

### Must Fix Before Merge:
1. ✅ **CRITICAL:** Make usage_history optional in RunContext (breaking change)
2. ✅ **CRITICAL:** Fix add_entry() race condition
3. ✅ **CRITICAL:** Fix cleanup_old_entries() race condition
4. ✅ **HIGH:** Fix all test failures
5. ✅ **HIGH:** Update TemporalRunContext

### Should Fix:
6. Lock deep copy issue
7. Add concurrent request tests
8. Add multi-agent integration tests
9. Documentation

### Nice to Have:
10. Consistent naming
11. Migration guide
12. Better error messages

---

## Estimated Work

**Critical Fixes:** 2-3 hours
**Tests:** 2-3 hours
**Documentation:** 1-2 hours

**Total:** ~8 hours to production ready

---

## Recommendation

**DO NOT MERGE** until critical issues #1-5 are fixed. The current implementation has:
- Breaking API changes
- Multiple race conditions that could cause data corruption
- Test failures preventing CI/CD

The prototype is conceptually sound but needs these fixes before it's safe to use.
