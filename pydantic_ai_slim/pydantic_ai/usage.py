from __future__ import annotations as _annotations

import dataclasses
import time
from copy import copy
from dataclasses import dataclass, field, fields
from typing import Annotated, Any

from genai_prices.data_snapshot import get_snapshot
from pydantic import AliasChoices, BeforeValidator, Field
from typing_extensions import deprecated, overload

from . import _utils
from .exceptions import UsageLimitExceeded

__all__ = 'RequestUsage', 'RunUsage', 'Usage', 'UsageLimits', 'UsageHistory', 'TimedUsageEntry'


@dataclass(repr=False, kw_only=True)
class UsageBase:
    input_tokens: Annotated[
        int,
        # `request_tokens` is deprecated, but we still want to support deserializing model responses stored in a DB before the name was changed
        Field(validation_alias=AliasChoices('input_tokens', 'request_tokens')),
    ] = 0
    """Number of input/prompt tokens."""

    cache_write_tokens: int = 0
    """Number of tokens written to the cache."""
    cache_read_tokens: int = 0
    """Number of tokens read from the cache."""

    output_tokens: Annotated[
        int,
        # `response_tokens` is deprecated, but we still want to support deserializing model responses stored in a DB before the name was changed
        Field(validation_alias=AliasChoices('output_tokens', 'response_tokens')),
    ] = 0
    """Number of output/completion tokens."""

    input_audio_tokens: int = 0
    """Number of audio input tokens."""
    cache_audio_read_tokens: int = 0
    """Number of audio tokens read from the cache."""
    output_audio_tokens: int = 0
    """Number of audio output tokens."""

    details: Annotated[
        dict[str, int],
        # `details` can not be `None` any longer, but we still want to support deserializing model responses stored in a DB before this was changed
        BeforeValidator(lambda d: d or {}),
    ] = dataclasses.field(default_factory=dict)
    """Any extra details returned by the model."""

    @property
    @deprecated('`request_tokens` is deprecated, use `input_tokens` instead')
    def request_tokens(self) -> int:
        return self.input_tokens

    @property
    @deprecated('`response_tokens` is deprecated, use `output_tokens` instead')
    def response_tokens(self) -> int:
        return self.output_tokens

    @property
    def total_tokens(self) -> int:
        """Sum of `input_tokens + output_tokens`."""
        return self.input_tokens + self.output_tokens

    def opentelemetry_attributes(self) -> dict[str, int]:
        """Get the token usage values as OpenTelemetry attributes."""
        result: dict[str, int] = {}
        if self.input_tokens:
            result['gen_ai.usage.input_tokens'] = self.input_tokens
        if self.output_tokens:
            result['gen_ai.usage.output_tokens'] = self.output_tokens

        details = self.details.copy()
        if self.cache_write_tokens:
            details['cache_write_tokens'] = self.cache_write_tokens
        if self.cache_read_tokens:
            details['cache_read_tokens'] = self.cache_read_tokens
        if self.input_audio_tokens:
            details['input_audio_tokens'] = self.input_audio_tokens
        if self.cache_audio_read_tokens:
            details['cache_audio_read_tokens'] = self.cache_audio_read_tokens
        if self.output_audio_tokens:
            details['output_audio_tokens'] = self.output_audio_tokens
        if details:
            prefix = 'gen_ai.usage.details.'
            for key, value in details.items():
                # Skipping check for value since spec implies all detail values are relevant
                if value:
                    result[prefix + key] = value
        return result

    def __repr__(self):
        kv_pairs = (f'{f.name}={value!r}' for f in fields(self) if (value := getattr(self, f.name)))
        return f'{self.__class__.__qualname__}({", ".join(kv_pairs)})'

    def has_values(self) -> bool:
        """Whether any values are set and non-zero."""
        return any(dataclasses.asdict(self).values())


@dataclass(repr=False, kw_only=True)
class RequestUsage(UsageBase):
    """LLM usage associated with a single request.

    This is an implementation of `genai_prices.types.AbstractUsage` so it can be used to calculate the price of the
    request using [genai-prices](https://github.com/pydantic/genai-prices).
    """

    @property
    def requests(self):
        return 1

    def incr(self, incr_usage: RequestUsage) -> None:
        """Increment the usage in place.

        Args:
            incr_usage: The usage to increment by.
        """
        return _incr_usage_tokens(self, incr_usage)

    def __add__(self, other: RequestUsage) -> RequestUsage:
        """Add two RequestUsages together.

        This is provided so it's trivial to sum usage information from multiple parts of a response.

        **WARNING:** this CANNOT be used to sum multiple requests without breaking some pricing calculations.
        """
        new_usage = copy(self)
        new_usage.incr(other)
        return new_usage

    @classmethod
    def extract(
        cls,
        data: Any,
        *,
        provider: str,
        provider_url: str,
        provider_fallback: str,
        api_flavor: str = 'default',
        details: dict[str, Any] | None = None,
    ) -> RequestUsage:
        """Extract usage information from the response data using genai-prices.

        Args:
            data: The response data from the model API.
            provider: The actual provider ID
            provider_url: The provider base_url
            provider_fallback: The fallback provider ID to use if the actual provider is not found in genai-prices.
                For example, an OpenAI model should set this to "openai" in case it has an obscure provider ID.
            api_flavor: The API flavor to use when extracting usage information,
                e.g. 'chat' or 'responses' for OpenAI.
            details: Becomes the `details` field on the returned `RequestUsage` for convenience.
        """
        details = details or {}
        for provider_id, provider_api_url in [(None, provider_url), (provider, None), (provider_fallback, None)]:
            try:
                provider_obj = get_snapshot().find_provider(None, provider_id, provider_api_url)
                _model_ref, extracted_usage = provider_obj.extract_usage(data, api_flavor=api_flavor)
                return cls(**{k: v for k, v in extracted_usage.__dict__.items() if v is not None}, details=details)
            except Exception:
                pass
        return cls(details=details)


@dataclass(repr=False, kw_only=True)
class RunUsage(UsageBase):
    """LLM usage associated with an agent run.

    Responsibility for calculating request usage is on the model; Pydantic AI simply sums the usage information across requests.
    """

    requests: int = 0
    """Number of requests made to the LLM API."""

    tool_calls: int = 0
    """Number of successful tool calls executed during the run."""

    input_tokens: int = 0
    """Total number of input/prompt tokens."""

    cache_write_tokens: int = 0
    """Total number of tokens written to the cache."""

    cache_read_tokens: int = 0
    """Total number of tokens read from the cache."""

    input_audio_tokens: int = 0
    """Total number of audio input tokens."""

    cache_audio_read_tokens: int = 0
    """Total number of audio tokens read from the cache."""

    output_tokens: int = 0
    """Total number of output/completion tokens."""

    details: dict[str, int] = dataclasses.field(default_factory=dict)
    """Any extra details returned by the model."""

    def incr(self, incr_usage: RunUsage | RequestUsage) -> None:
        """Increment the usage in place.

        Args:
            incr_usage: The usage to increment by.
        """
        if isinstance(incr_usage, RunUsage):
            self.requests += incr_usage.requests
            self.tool_calls += incr_usage.tool_calls
        return _incr_usage_tokens(self, incr_usage)

    def __add__(self, other: RunUsage | RequestUsage) -> RunUsage:
        """Add two RunUsages together.

        This is provided so it's trivial to sum usage information from multiple runs.
        """
        new_usage = copy(self)
        new_usage.incr(other)
        return new_usage


def _incr_usage_tokens(slf: RunUsage | RequestUsage, incr_usage: RunUsage | RequestUsage) -> None:
    """Increment the usage in place.

    Args:
        slf: The usage to increment.
        incr_usage: The usage to increment by.
    """
    slf.input_tokens += incr_usage.input_tokens
    slf.cache_write_tokens += incr_usage.cache_write_tokens
    slf.cache_read_tokens += incr_usage.cache_read_tokens
    slf.input_audio_tokens += incr_usage.input_audio_tokens
    slf.cache_audio_read_tokens += incr_usage.cache_audio_read_tokens
    slf.output_tokens += incr_usage.output_tokens

    for key, value in incr_usage.details.items():
        slf.details[key] = slf.details.get(key, 0) + value


@dataclass(repr=False, kw_only=True)
@deprecated('`Usage` is deprecated, use `RunUsage` instead')
class Usage(RunUsage):
    """Deprecated alias for `RunUsage`."""


@dataclass
class TimedUsageEntry:
    """A single usage entry with timestamp for rate limiting.

    This class tracks when a request was made and its associated usage,
    enabling sliding window rate limit calculations.
    """

    timestamp: float
    """Unix timestamp when this usage occurred."""

    usage: RequestUsage
    """The usage information for this request."""


@dataclass
class UsageHistory:
    """Tracks usage history for per-minute rate limiting.

    This class maintains a history of usage entries with timestamps,
    allowing calculation of usage within sliding time windows (e.g., last 60 seconds).
    """

    entries: list[TimedUsageEntry] = field(default_factory=list)
    """List of usage entries with timestamps."""

    def add_entry(self, usage: RequestUsage, timestamp: float | None = None) -> None:
        """Add a new usage entry with timestamp.

        Args:
            usage: The usage information to record.
            timestamp: Optional explicit timestamp. If None, uses current time.
        """
        if timestamp is None:
            timestamp = time.time()
        self.entries.append(TimedUsageEntry(timestamp=timestamp, usage=usage))

    def get_usage_in_window(self, window_seconds: float) -> RunUsage:
        """Calculate total usage within the last N seconds.

        Args:
            window_seconds: The size of the time window in seconds (e.g., 60.0 for last minute).

        Returns:
            Aggregated usage from all entries within the time window.
        """
        cutoff = time.time() - window_seconds

        # Sum up usage from entries within the window
        total = RunUsage()
        requests = 0
        for entry in self.entries:
            if entry.timestamp >= cutoff:
                total.incr(entry.usage)
                requests += 1

        # Set request count based on number of entries in window
        total.requests = requests
        return total

    def cleanup_old_entries(self, window_seconds: float) -> None:
        """Remove entries older than the specified window to prevent memory growth.

        Args:
            window_seconds: Entries older than this will be removed.
        """
        cutoff = time.time() - window_seconds
        self.entries = [e for e in self.entries if e.timestamp >= cutoff]


@dataclass(repr=False, kw_only=True)
class UsageLimits:
    """Limits on model usage.

    The request count is tracked by pydantic_ai, and the request limit is checked before each request to the model.
    Token counts are provided in responses from the model, and the token limits are checked after each response.

    Each of the limits can be set to `None` to disable that limit.
    """

    request_limit: int | None = 50
    """The maximum number of requests allowed to the model."""
    tool_calls_limit: int | None = None
    """The maximum number of successful tool calls allowed to be executed."""
    input_tokens_limit: int | None = None
    """The maximum number of input/prompt tokens allowed."""
    output_tokens_limit: int | None = None
    """The maximum number of output/response tokens allowed."""
    total_tokens_limit: int | None = None
    """The maximum number of tokens allowed in requests and responses combined."""
    count_tokens_before_request: bool = False
    """If True, perform a token counting pass before sending the request to the model,
    to enforce `request_tokens_limit` ahead of time. This may incur additional overhead
    (from calling the model's `count_tokens` API before making the actual request) and is disabled by default."""

    # Per-minute rate limits (sliding window)
    requests_per_minute: int | None = None
    """The maximum number of requests allowed per minute (60-second sliding window)."""
    input_tokens_per_minute: int | None = None
    """The maximum number of input tokens allowed per minute (60-second sliding window)."""
    output_tokens_per_minute: int | None = None
    """The maximum number of output tokens allowed per minute (60-second sliding window)."""
    total_tokens_per_minute: int | None = None
    """The maximum total tokens (input + output) allowed per minute (60-second sliding window)."""

    @property
    @deprecated('`request_tokens_limit` is deprecated, use `input_tokens_limit` instead')
    def request_tokens_limit(self) -> int | None:
        return self.input_tokens_limit

    @property
    @deprecated('`response_tokens_limit` is deprecated, use `output_tokens_limit` instead')
    def response_tokens_limit(self) -> int | None:
        return self.output_tokens_limit

    @overload
    def __init__(
        self,
        *,
        request_limit: int | None = 50,
        tool_calls_limit: int | None = None,
        input_tokens_limit: int | None = None,
        output_tokens_limit: int | None = None,
        total_tokens_limit: int | None = None,
        count_tokens_before_request: bool = False,
        requests_per_minute: int | None = None,
        input_tokens_per_minute: int | None = None,
        output_tokens_per_minute: int | None = None,
        total_tokens_per_minute: int | None = None,
    ) -> None:
        self.request_limit = request_limit
        self.tool_calls_limit = tool_calls_limit
        self.input_tokens_limit = input_tokens_limit
        self.output_tokens_limit = output_tokens_limit
        self.total_tokens_limit = total_tokens_limit
        self.count_tokens_before_request = count_tokens_before_request
        self.requests_per_minute = requests_per_minute
        self.input_tokens_per_minute = input_tokens_per_minute
        self.output_tokens_per_minute = output_tokens_per_minute
        self.total_tokens_per_minute = total_tokens_per_minute

    @overload
    @deprecated(
        'Use `input_tokens_limit` instead of `request_tokens_limit` and `output_tokens_limit` and `total_tokens_limit`'
    )
    def __init__(
        self,
        *,
        request_limit: int | None = 50,
        tool_calls_limit: int | None = None,
        request_tokens_limit: int | None = None,
        response_tokens_limit: int | None = None,
        total_tokens_limit: int | None = None,
        count_tokens_before_request: bool = False,
        requests_per_minute: int | None = None,
        input_tokens_per_minute: int | None = None,
        output_tokens_per_minute: int | None = None,
        total_tokens_per_minute: int | None = None,
    ) -> None:
        self.request_limit = request_limit
        self.tool_calls_limit = tool_calls_limit
        self.input_tokens_limit = request_tokens_limit
        self.output_tokens_limit = response_tokens_limit
        self.total_tokens_limit = total_tokens_limit
        self.count_tokens_before_request = count_tokens_before_request
        self.requests_per_minute = requests_per_minute
        self.input_tokens_per_minute = input_tokens_per_minute
        self.output_tokens_per_minute = output_tokens_per_minute
        self.total_tokens_per_minute = total_tokens_per_minute

    def __init__(
        self,
        *,
        request_limit: int | None = 50,
        tool_calls_limit: int | None = None,
        input_tokens_limit: int | None = None,
        output_tokens_limit: int | None = None,
        total_tokens_limit: int | None = None,
        count_tokens_before_request: bool = False,
        requests_per_minute: int | None = None,
        input_tokens_per_minute: int | None = None,
        output_tokens_per_minute: int | None = None,
        total_tokens_per_minute: int | None = None,
        # deprecated:
        request_tokens_limit: int | None = None,
        response_tokens_limit: int | None = None,
    ):
        self.request_limit = request_limit
        self.tool_calls_limit = tool_calls_limit
        self.input_tokens_limit = input_tokens_limit or request_tokens_limit
        self.output_tokens_limit = output_tokens_limit or response_tokens_limit
        self.total_tokens_limit = total_tokens_limit
        self.count_tokens_before_request = count_tokens_before_request
        self.requests_per_minute = requests_per_minute
        self.input_tokens_per_minute = input_tokens_per_minute
        self.output_tokens_per_minute = output_tokens_per_minute
        self.total_tokens_per_minute = total_tokens_per_minute

    def has_token_limits(self) -> bool:
        """Returns `True` if this instance places any limits on token counts.

        If this returns `False`, the `check_tokens` method will never raise an error.

        This is useful because if we have token limits, we need to check them after receiving each streamed message.
        If there are no limits, we can skip that processing in the streaming response iterator.
        """
        return any(
            limit is not None for limit in (self.input_tokens_limit, self.output_tokens_limit, self.total_tokens_limit)
        )

    def check_before_request(self, usage: RunUsage) -> None:
        """Raises a `UsageLimitExceeded` exception if the next request would exceed any of the limits."""
        request_limit = self.request_limit
        if request_limit is not None and usage.requests >= request_limit:
            raise UsageLimitExceeded(f'The next request would exceed the request_limit of {request_limit}')

        input_tokens = usage.input_tokens
        if self.input_tokens_limit is not None and input_tokens > self.input_tokens_limit:
            raise UsageLimitExceeded(
                f'The next request would exceed the input_tokens_limit of {self.input_tokens_limit} ({input_tokens=})'
            )

        total_tokens = usage.total_tokens
        if self.total_tokens_limit is not None and total_tokens > self.total_tokens_limit:
            raise UsageLimitExceeded(  # pragma: lax no cover
                f'The next request would exceed the total_tokens_limit of {self.total_tokens_limit} ({total_tokens=})'
            )

    def check_tokens(self, usage: RunUsage) -> None:
        """Raises a `UsageLimitExceeded` exception if the usage exceeds any of the token limits."""
        input_tokens = usage.input_tokens
        if self.input_tokens_limit is not None and input_tokens > self.input_tokens_limit:
            raise UsageLimitExceeded(f'Exceeded the input_tokens_limit of {self.input_tokens_limit} ({input_tokens=})')

        output_tokens = usage.output_tokens
        if self.output_tokens_limit is not None and output_tokens > self.output_tokens_limit:
            raise UsageLimitExceeded(
                f'Exceeded the output_tokens_limit of {self.output_tokens_limit} ({output_tokens=})'
            )

        total_tokens = usage.total_tokens
        if self.total_tokens_limit is not None and total_tokens > self.total_tokens_limit:
            raise UsageLimitExceeded(f'Exceeded the total_tokens_limit of {self.total_tokens_limit} ({total_tokens=})')

    def check_before_tool_call(self, projected_usage: RunUsage) -> None:
        """Raises a `UsageLimitExceeded` exception if the next tool call(s) would exceed the tool call limit."""
        tool_calls_limit = self.tool_calls_limit
        tool_calls = projected_usage.tool_calls
        if tool_calls_limit is not None and tool_calls > tool_calls_limit:
            raise UsageLimitExceeded(
                f'The next tool call(s) would exceed the tool_calls_limit of {tool_calls_limit} ({tool_calls=}).'
            )

    def has_per_minute_limits(self) -> bool:
        """Returns `True` if this instance places any per-minute limits.

        If this returns `False`, the per-minute checking methods will never raise an error.
        """
        return any(
            limit is not None
            for limit in (
                self.requests_per_minute,
                self.input_tokens_per_minute,
                self.output_tokens_per_minute,
                self.total_tokens_per_minute,
            )
        )

    def check_per_minute_limits_before_request(
        self, usage_history: UsageHistory, projected_input_tokens: int = 0
    ) -> None:
        """Check if making a request would exceed per-minute limits.

        Args:
            usage_history: The usage history to check against.
            projected_input_tokens: The expected input tokens for the next request (if known).

        Raises:
            UsageLimitExceeded: If the next request would exceed any per-minute limit.
        """
        if not self.has_per_minute_limits():
            return

        # Get usage in last 60 seconds
        window_usage = usage_history.get_usage_in_window(60.0)

        # Check requests per minute
        if self.requests_per_minute is not None:
            if window_usage.requests + 1 > self.requests_per_minute:
                raise UsageLimitExceeded(
                    f'The next request would exceed the requests_per_minute limit of {self.requests_per_minute} '
                    f'(requests_in_last_minute={window_usage.requests})'
                )

        # Check input tokens per minute (if we have a projection)
        if projected_input_tokens > 0 and self.input_tokens_per_minute is not None:
            projected_input = window_usage.input_tokens + projected_input_tokens
            if projected_input > self.input_tokens_per_minute:
                raise UsageLimitExceeded(
                    f'The next request would exceed the input_tokens_per_minute limit of {self.input_tokens_per_minute} '
                    f'(projected_input_tokens_in_last_minute={projected_input})'
                )

        # Check total tokens per minute (if we have a projection)
        if projected_input_tokens > 0 and self.total_tokens_per_minute is not None:
            projected_total = window_usage.total_tokens + projected_input_tokens
            if projected_total > self.total_tokens_per_minute:
                raise UsageLimitExceeded(
                    f'The next request would exceed the total_tokens_per_minute limit of {self.total_tokens_per_minute} '
                    f'(projected_total_tokens_in_last_minute={projected_total})'
                )

    def check_per_minute_limits_after_response(self, usage_history: UsageHistory) -> None:
        """Check if actual usage exceeded per-minute limits.

        Args:
            usage_history: The usage history to check against.

        Raises:
            UsageLimitExceeded: If actual usage exceeded any per-minute limit.
        """
        if not self.has_per_minute_limits():
            return

        window_usage = usage_history.get_usage_in_window(60.0)

        # Check input tokens per minute
        if self.input_tokens_per_minute is not None and window_usage.input_tokens > self.input_tokens_per_minute:
            raise UsageLimitExceeded(
                f'Exceeded the input_tokens_per_minute limit of {self.input_tokens_per_minute} '
                f'(input_tokens_in_last_minute={window_usage.input_tokens})'
            )

        # Check output tokens per minute
        if self.output_tokens_per_minute is not None and window_usage.output_tokens > self.output_tokens_per_minute:
            raise UsageLimitExceeded(
                f'Exceeded the output_tokens_per_minute limit of {self.output_tokens_per_minute} '
                f'(output_tokens_in_last_minute={window_usage.output_tokens})'
            )

        # Check total tokens per minute
        if self.total_tokens_per_minute is not None and window_usage.total_tokens > self.total_tokens_per_minute:
            raise UsageLimitExceeded(
                f'Exceeded the total_tokens_per_minute limit of {self.total_tokens_per_minute} '
                f'(total_tokens_in_last_minute={window_usage.total_tokens})'
            )

    __repr__ = _utils.dataclasses_no_defaults_repr
