# Bug Report: Per-Minute Rate Limiting Prototype

## Critical Bugs

### 1. **CRITICAL: Multi-Agent Scenario - UsageHistory Not Propagated**

**Location:** `pydantic_ai_slim/pydantic_ai/agent/__init__.py:575`

**Issue:** When an agent passes `usage` to another agent (multi-agent scenarios), the `usage_history` is NOT propagated along with it. This completely breaks per-minute rate limiting across agents.

**Current Code:**
```python
# Line 574-580
usage = usage or _usage.RunUsage()
state = _agent_graph.GraphAgentState(
    message_history=list(message_history) if message_history else [],
    usage=usage,  # ✅ Passed
    retries=0,
    run_step=0,
)
# ❌ usage_history is NOT passed, so it defaults to empty UsageHistory()
```

**Impact:**
- Per-minute rate limits are reset when delegating to another agent
- Users can bypass RPM/TPM limits by using multiple agents
- Rate limiting only works within a single agent run

**Example Failure:**
```python
controller_agent = Agent('bedrock:anthropic.claude-v2')

@controller_agent.tool
async def delegate(ctx: RunContext[None], query: str) -> str:
    # Pass usage to accumulate token counts (existing behavior)
    result = await delegate_agent.run(query, usage=ctx.usage)
    return result.output

# First request uses 90% of RPM limit
# Delegate to another agent - usage_history is LOST
# Second request in delegate can exceed RPM because history is empty!
```

**Fix Required:**
1. Add `usage_history` parameter to agent `run()`, `run_sync()`, and `iter()` methods
2. Pass `usage_history` when creating `GraphAgentState`
3. Update `RunContext` to expose `usage_history` so tools can pass it

---

### 2. **BUG: Streaming - UsageHistory Not Checked**

**Location:** `pydantic_ai_slim/pydantic_ai/result.py:729-743`

**Issue:** The streaming response checker `_get_usage_checking_stream_response()` does NOT receive or check `usage_history`, so per-minute limits are never enforced during streaming.

**Current Code:**
```python
def _get_usage_checking_stream_response(
    stream_response: models.StreamedResponse,
    limits: UsageLimits | None,
    get_usage: Callable[[], RunUsage],  # ❌ No usage_history parameter
) -> AsyncIterator[ModelResponseStreamEvent]:
    if limits is not None and limits.has_token_limits():
        async def _usage_checking_iterator():
            async for item in stream_response:
                limits.check_tokens(get_usage())  # ✅ Checks cumulative
                # ❌ Never checks per-minute limits!
                yield item
        return _usage_checking_iterator()
```

**Impact:**
- Per-minute limits are completely ignored during streaming
- Users can bypass TPM limits by using streaming responses
- Only cumulative limits work in streaming mode

**Fix Required:**
1. Add `usage_history: UsageHistory | None` parameter to `_get_usage_checking_stream_response()`
2. Check per-minute limits in the iterator
3. Pass `usage_history` from `AgentStream.__aiter__()` (line 269)

---

### 3. **BUG: Race Condition - Concurrent Limit Checks**

**Location:** `pydantic_ai_slim/pydantic_ai/usage.py:289-310`

**Issue:** `UsageHistory` is not thread-safe. If multiple coroutines check limits concurrently, they may all see the same usage and all pass the check, then all add entries, exceeding the limit.

**Scenario:**
```python
# Three concurrent requests at 99 RPM (limit: 100)
# All three check usage_history simultaneously
# All three see: 99 requests in window (< 100) ✅
# All three proceed and add entries
# Result: 102 requests in window (> 100) ❌
```

**Impact:**
- Rate limits can be exceeded by up to N-1 requests where N is concurrency
- More likely with tool calls (which run in parallel)
- Could cause API rate limit errors from providers

**Fix Required:**
Use `asyncio.Lock` to make `check_per_minute_limits_before_request()` and `add_entry()` atomic:
```python
@dataclass
class UsageHistory:
    entries: list[TimedUsageEntry] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def add_entry_atomic(self, usage: RequestUsage) -> None:
        async with self._lock:
            self.entries.append(...)
```

---

## Medium Severity Bugs

### 4. **BUG: Memory Leak - Cleanup Not Guaranteed**

**Location:** `pydantic_ai_slim/pydantic_ai/_agent_graph.py:499-500`

**Issue:** `cleanup_old_entries()` is only called IF per-minute limits are enabled. If a user sets limits, accumulates history, then removes limits, the history grows unbounded.

**Current Code:**
```python
# Clean up old usage history entries to prevent memory growth
if ctx.deps.usage_limits.has_per_minute_limits():
    ctx.state.usage_history.cleanup_old_entries(60.0)
```

**Impact:**
- Memory leak if limits are toggled
- In long-running agents, history could grow very large
- Minor issue but worth fixing

**Fix:**
Always cleanup regardless of whether limits are currently enabled:
```python
# Always cleanup to prevent memory leaks
ctx.state.usage_history.cleanup_old_entries(60.0)
```

---

### 5. **BUG: Serialization - UsageHistory Not Serializable**

**Location:** `pydantic_ai_slim/pydantic_ai/usage.py:264-319`

**Issue:** `UsageHistory` and `TimedUsageEntry` are plain `@dataclass` without Pydantic support, so they cannot be serialized/deserialized. This breaks state persistence.

**Impact:**
- Cannot save/restore agent state with per-minute limits
- Breaks durable execution scenarios
- Integration with state stores (Redis, DB) will fail

**Fix:**
Convert to Pydantic models or add explicit serialization support:
```python
from pydantic import BaseModel

class TimedUsageEntry(BaseModel):
    timestamp: float
    usage: RequestUsage

class UsageHistory(BaseModel):
    entries: list[TimedUsageEntry] = []
```

---

## Minor Issues

### 6. **MINOR: Unclear Error Messages**

**Location:** `pydantic_ai_slim/pydantic_ai/usage.py:536-594`

**Issue:** Error messages say "requests_in_last_minute" but the value shown is the count BEFORE the current request, which is confusing.

**Example:**
```
The next request would exceed the requests_per_minute limit of 100 (requests_in_last_minute=99)
```

User thinks: "99 < 100, why is it failing?"
Reality: It's checking if 99 + 1 > 100

**Fix:**
Make messages clearer:
```python
raise UsageLimitExceeded(
    f'The next request would exceed the requests_per_minute limit of {self.requests_per_minute} '
    f'(current: {window_usage.requests}, would become: {window_usage.requests + 1})'
)
```

---

### 7. **MINOR: No Window Size Configuration**

**Location:** Hardcoded `60.0` throughout code

**Issue:** The 60-second window is hardcoded everywhere. Some providers may use different windows (e.g., 10 seconds, 5 minutes).

**Impact:**
- Cannot adapt to different provider rate limit windows
- Minor limitation but worth noting

**Fix (Future Enhancement):**
Add configurable window size to `UsageLimits`:
```python
class UsageLimits:
    requests_per_minute: int | None = None
    rate_limit_window_seconds: float = 60.0  # Configurable window
```

---

### 8. **MINOR: Missing Type Validation**

**Location:** `pydantic_ai_slim/pydantic_ai/usage.py:345-351`

**Issue:** No validation that per-minute limits are positive integers.

**Impact:**
- User could set negative limits or zero limits
- Could cause confusing behavior

**Fix:**
Add validation in `__post_init__`:
```python
def __post_init__(self):
    for field_name in ['requests_per_minute', 'input_tokens_per_minute', ...]:
        value = getattr(self, field_name)
        if value is not None and value < 0:
            raise ValueError(f'{field_name} must be non-negative, got {value}')
```

---

## Edge Cases to Test

### 9. **EDGE: Empty History**
- ✅ Currently handles correctly (returns empty RunUsage)

### 10. **EDGE: All Entries Outside Window**
- ✅ Currently handles correctly (returns empty RunUsage)

### 11. **EDGE: Monotonic Clock Overflow**
- ⚠️ Extremely unlikely but theoretically possible
- `time.monotonic()` could overflow after ~292 billion years on 64-bit systems
- Not a practical concern

### 12. **EDGE: Negative Window**
- ❌ No validation - would return empty results
- Should add validation

---

## Summary

### Must Fix Before Production:
1. ✅ Monotonic clock (FIXED)
2. ❌ Multi-agent usage_history propagation (CRITICAL)
3. ❌ Streaming per-minute checks (CRITICAL)
4. ❌ Race condition / thread safety (HIGH)

### Should Fix:
5. Memory leak on cleanup
6. Serialization support
7. Clearer error messages

### Nice to Have:
8. Configurable window size
9. Input validation
10. Edge case hardening

---

## Testing Gaps

Current tests DO NOT cover:
- ❌ Multi-agent scenarios with per-minute limits
- ❌ Streaming with per-minute limits
- ❌ Concurrent requests hitting limits
- ❌ Serialization/deserialization
- ❌ Long-running agents with cleanup
- ❌ Edge cases (negative windows, empty history, etc.)

---

## Recommendations

1. **Immediate:** Fix critical bugs #1, #2, #3
2. **Before merge:** Add comprehensive integration tests
3. **Documentation:** Add warning about current limitations
4. **Future:** Consider using a proper rate limiting library (e.g., `aiolimiter`) instead of rolling our own
