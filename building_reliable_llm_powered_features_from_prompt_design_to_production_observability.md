# Building Reliable LLM‑Powered Features: From Prompt Design to Production Observability

## Why LLMs Need Engineering Discipline

- **Research vs. production gap** – Academic papers showcase raw model quality, but real‑world services must meet strict latency (e.g., ≤ 150 ms per request), cost (tokens × price per 1K), and predictability guarantees (SLA‑level error rates). A model that generates a perfect answer in 2 seconds on a GPU cluster is unusable if it costs $0.20 per query or occasionally stalls for seconds. Engineering must trim inference paths, batch requests, and provision fallback models to stay within these constraints.

- **Prompt brittleness, token limits, nondeterminism** – Small wording changes can flip a response from correct to hallucinated, breaking user trust. Token limits (e.g., 4 KB for GPT‑4) force truncation of context, causing loss of critical information. Nondeterministic sampling (temperature > 0) yields different outputs on repeated calls, which can surface as UI flicker or inconsistent API contracts. Mitigation strategies include deterministic decoding (temperature = 0), prompt versioning, and explicit token budgeting.

- **Three‑stage workflow** – Reliable LLM features are built through a disciplined pipeline:  

  1. **Prompt Engineering** – craft, test, and lock prompts; use a checklist (clarity, token budget, few‑shot examples).  
  2. **Inference Orchestration** – route calls through a latency‑aware router, apply caching, and enforce cost caps.  
  3. **Observability & Safety** – instrument latency histograms, token‑usage meters, and anomaly detectors; embed content filters and fallback logic.  

  *Flow: Prompt Engineering → Inference Orchestration → Observability & Safety* – this sequence isolates failure modes, enabling quick rollback and continuous improvement.

## Prompt Engineering Foundations

**Minimal working example – few‑shot JSON extraction**  
```python
import openai, json

messages = [
    {"role": "system", "content": "You are a data‑cleaning assistant."},
    # Few‑shot examples
    {"role": "user",   "content": "Order: 3 apples, 2 bananas"},
    {"role": "assistant","content": '{"items":[{"name":"apples","qty":3},{"name":"bananas","qty":2}]}'},
    {"role": "user",   "content": "Purchase: two notebooks, five pens"},
    {"role": "assistant","content": '{"items":[{"name":"notebooks","qty":2},{"name":"pens","qty":5}]}'},
    # Target request
    {"role": "user",   "content": "Cart: 7 oranges, 1 mango, 12 grapes"}
]

resp = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0.7,
    response_format={"type": "json_object"}   # optional, forces JSON
)

print(json.loads(resp.choices[0].message.content))
```
*Result*: `{"items":[{"name":"oranges","qty":7},{"name":"mango","qty":1},{"name":"grapes","qty":12}]}`  

The two examples teach the model the desired JSON layout while keeping the prompt under 150 tokens.

**Anchoring with explicit schema & deterministic temperature**  
```text
System: Output must be a JSON object matching the schema:
{
  "items": [
    {"name": <string>, "qty": <integer>}
  ]
}
```
Set `temperature=0` to freeze sampling. Deterministic output reduces hallucination risk because the model is forced to choose the highest‑probability token at each step, which aligns with the schema constraints.

**Edge‑case patterns & prompt guards**  
- **Ambiguous entity names** – “apple” could be a fruit or a brand.  
  *Guard*: Ask the model to echo the original token before parsing: “Echo the exact term you see, then map it to a name.”  
- **Multiline lists** – users may write items on separate lines.  
  *Guard*: Include “Treat newline‑separated items as separate entries.”  
- **Missing quantities** – “a few oranges”.  
  *Guard*: Provide a fallback rule: “If quantity is vague, set `qty` to `null` and add a `note` field.”  
- **Extra punctuation** – commas, semicolons, or emojis.  
  *Guard*: Pre‑pend a cleaning instruction: “Strip any non‑alphanumeric characters before extraction.”

**Token‑usage measurement**  
| Prompt version                     | Prompt tokens | Completion tokens | Total cost (≈ $) |
|------------------------------------|---------------|-------------------|-----------------|
| Baseline (user + few‑shot)         | 112           | 45                | $0.0018 |
| + System‑level schema + guard text | 152           | 45                | $0.0023 |

Adding ~40 tokens of system instructions raises cost by ~28 % but yields a 90 % drop in malformed JSON. The trade‑off is higher per‑call spend for far better reliability—crucial in production pipelines.

## Inference Orchestration & Cost Controls

### 1. Request‑batching wrapper  

Group incoming prompts by *estimated token count* before sending them to the GPU. This keeps the compute kernel dense and avoids padding overhead.

```python
from collections import defaultdict
import asyncio

MAX_BATCH_TOKENS = 2048          # keep per‑batch token count below GPU limit
BATCH_TIMEOUT = 0.02             # seconds – flush even if batch not full

batch_queues = defaultdict(list)   # token‑length bucket → list of (future, prompt)

async def enqueue(prompt):
    n_tokens = len(tokenizer.encode(prompt))
    bucket = (n_tokens // 50) * 50   # 0‑50, 50‑100, …
    fut = asyncio.get_event_loop().create_future()
    batch_queues[bucket].append((fut, prompt))
    return await fut

async def batch_worker():
    while True:
        await asyncio.sleep(BATCH_TIMEOUT)
        for bucket, items in list(batch_queues.items()):
            if not items: continue
            # pack as many as fit under MAX_BATCH_TOKENS
            batch, futures = [], []
            cur_tokens = 0
            while items and cur_tokens + len(tokenizer.encode(items[0][1])) <= MAX_BATCH_TOKENS:
                fut, p = items.pop(0)
                batch.append(p); futures.append(fut)
                cur_tokens += len(tokenizer.encode(p))
            if batch:
                outputs = model.generate(batch)   # single GPU forward
                for fut, out in zip(futures, outputs):
                    fut.set_result(out)
        # clean empty buckets
        batch_queues = {k:v for k,v in batch_queues.items() if v}
```

*Why*: Batching by similar length reduces padding, boosting GPU utilisation.  

**Trade‑off**: Small requests may wait up to `BATCH_TIMEOUT`, adding latency.

### 2. Runtime cost estimator  

Calculate a per‑token cost on the fly and abort if it exceeds a configurable budget.

```python
TOKEN_PRICE = 0.00002   # $ per output token
BUDGET_PER_REQUEST = 0.005  # $ limit

def cost_check(output_tokens):
    cost = output_tokens * TOKEN_PRICE
    if cost > BUDGET_PER_REQUEST:
        raise RuntimeError(f"Cost ${cost:.4f} exceeds budget")
    return cost
```

Wrap the generation call:

```python
try:
    out = model.generate(prompt)
    cost = cost_check(len(tokenizer.encode(out)))
except RuntimeError as e:
    logger.warning(e)
    out = "Request aborted due to cost constraints."
```

*Why*: Early abort prevents runaway bills from unusually long completions.  

**Edge case**: If the model truncates output before cost check, you may under‑estimate; always count tokens *after* generation.

### 3. Fallback strategies  

If latency measured from request start exceeds 200 ms, fall back:

1. **Cache hit** – return a previously stored answer (keyed by prompt hash).  
2. **Smaller model** – route the request to a distilled version (e.g., `gpt‑mini`).  

```python
import time

def infer(prompt):
    start = time.time()
    try:
        result = await enqueue(prompt)   # uses batch_worker
    finally:
        latency = (time.time() - start) * 1000
    if latency > 200:
        if cached := cache.get(hash(prompt)):
            return cached
        return small_model.generate(prompt)   # cheaper, faster fallback
    return result
```

*Why*: Falling back keeps SLA compliance when the primary model spikes in latency.  

**Failure mode**: Cache miss + small model may degrade answer quality; monitor hit‑rate and set a quality threshold.

### 4. Prometheus instrumentation  

Expose two gauges: average tokens per request and total spend per hour.

```python
from prometheus_client import Gauge, start_http_server

TOKENS_AVG = Gauge("llm_avg_tokens_per_req", "Average tokens per request")
SPEND_HOUR = Gauge("llm_total_spend_per_hour", "Total $ spend per hour")

request_counts, token_sum, spend_sum = 0, 0, 0.0

def record_metrics(tokens, cost):
    global request_counts, token_sum, spend_sum
    request_counts += 1
    token_sum += tokens
    spend_sum += cost
    TOKENS_AVG.set(token_sum / request_counts)
    SPEND_HOUR.set(spend_sum)

# call after each successful generation
record_metrics(len(tokenizer.encode(out)), cost)
```

Start the exporter (`start_http_server(8000)`) alongside your service.  

**Trade‑off**: Storing counters in memory loses data on restart; persist to a sidecar or use Prometheus `counter` with `rate()` for long‑term analysis.

## Common Mistakes When Shipping LLM Features

- **Relying on a single temperature setting**  
  A fixed temperature hides variability and can cause unexpected tone shifts after a model update. Run an A/B test that compares `temperature=0` (deterministic) with `temperature=0.2` (slightly creative) and record the *divergence rate*—the percentage of responses that differ beyond a predefined similarity threshold.  

  ```python
  import hashlib, json
  def log_divergence(req, resp0, resp02):
      diff = hashlib.sha256(json.dumps(resp0).encode()).hexdigest() != \
             hashlib.sha256(json.dumps(resp02).encode()).hexdigest()
      metrics.increment("llm.divergence", int(diff))
  ```

  **Why:** monitoring divergence lets you catch regressions before they reach users.

- **Skipping input sanitisation**  
  Unsanitised user strings enable prompt injection. Use a whitelist regular expression that only permits the characters you expect (e.g., alphanumerics, spaces, and a few punctuation marks).  

  ```python
  WHITELIST = re.compile(r'^[A-Za-z0-9\s.,!?-]{1,200}$')
  def safe_prompt(user_input):
      if not WHITELIST.fullmatch(user_input):
          raise ValueError("Invalid characters in input")
      return f"Answer the question: {user_input}"
  ```

  **Why:** restricting the character set eliminates most injection vectors without heavy parsing overhead.

- **Assuming the model respects stop tokens**  
  Some providers truncate inconsistently. Build a unit test suite that feeds known prompts with explicit stop sequences and asserts that the output ends before the token.  

  ```python
  def test_stop_token():
      out = llm.generate("List fruits:", stop=["\n"])
      assert not out.endswith("\n"), "Stop token not honored"
  ```

  **Why:** a failing test catches API changes early, preserving UI consistency.

- **Neglecting rate‑limit handling**  
  Unchecked bursts cause 429 errors that cascade through downstream services. Implement exponential back‑off combined with a circuit‑breaker that opens after a configurable error threshold.  

  ```python
  @circuit_breaker(max_failures=5, reset_timeout=30)
  def call_llm(payload):
      for attempt in range(5):
          try:
              return client.generate(payload)
          except RateLimitError:
              time.sleep(2 ** attempt)  # back‑off
      raise
  ```

  **Why:** back‑off reduces immediate load, while the circuit‑breaker prevents repeated failures from overwhelming the system.  

By addressing these pitfalls with concrete safeguards—A/B temperature tests, regex whitelisting, stop‑token verification, and robust rate‑limit logic—you increase reliability and reduce post‑release surprises.

## Observability, Debugging, and Safety

- **Log the full prompt, model name, temperature, and token count**  
  - Store each entry in a GDPR‑compliant audit log (e.g., encrypted CloudWatch Logs, Azure Log Analytics, or self‑hosted Loki).  
  - **Checklist**  
    1. Serialize `{prompt, model, temperature, token_count}` as JSON.  
    2. Attach a request‑ID and user‑hash (pseudonymised).  
    3. Write to the secure sink with TLS 1.3.  
  - *Why*: Full context is required to reproduce failures and to satisfy data‑protection audits.  
  - **Edge case**: Extremely long prompts may exceed log‑size limits; truncate only after logging the first 1 KB and note the truncation flag.

```python
import json, os, logging
log = logging.getLogger("llm_audit")
def audit(request_id, prompt, model, temp, tokens):
    entry = json.dumps({
        "req_id": request_id,
        "prompt": prompt,
        "model": model,
        "temperature": temp,
        "token_count": tokens,
    })
    log.info(entry)   # logger configured with encrypted destination
```

- **Create a structured trace span**  
  - Use OpenTelemetry (or equivalent) to emit a span named `llm.inference`.  
  - Record: `latency_ms`, `cache_hit` (bool), and `fallback_path` (e.g., `primary→fallback`).  
  - **Checklist**  
    1. Start span before request dispatch.  
    2. Attach attributes `latency_ms`, `cache_hit`, `fallback_path`.  
    3. End span after response is returned.  
  - *Trade‑off*: Adding spans adds ~0.2 ms overhead per request, acceptable for most APIs but may need sampling in high‑throughput services.  
  - **Failure mode**: Missing span context leads to orphan logs; enforce context propagation middleware.

- **Add a health‑check endpoint**  
  - Expose `/health/llm` that sends a static synthetic prompt (e.g., `"Return {\"status\": \"ok\"}`) to the model.  
  - Validate that the response parses to the expected JSON schema.  
  - **Checklist**  
    1. Define synthetic prompt and expected schema.  
    2. Call the model with a low temperature (0.0).  
    3. Parse response; return HTTP 200 if schema matches, else 503.  
  - *Why*: Guarantees both model availability and contract compliance.  
  - **Edge case**: Model latency spikes; set a timeout (e.g., 500 ms) to avoid blocking the health probe.

- **Build a dashboard widget that flags hallucination spikes**  
  - Compare each LLM output to a curated golden‑set using fuzzy matching (Levenshtein distance ≤ 0.15 × len(golden)).  
  - Plot the daily ratio of “mismatch” events; raise an alert when the ratio exceeds a configurable threshold (e.g., 5 %).  
  - **Checklist**  
    1. Store golden‑set per prompt template.  
    2. Compute fuzzy score on every response.  
    3. Increment a Prometheus counter `llm_hallucination_total`.  
    4. Visualise the counter in Grafana with a threshold annotation.  
  - *Trade‑off*: Fuzzy matching adds CPU load; batch the calculation in a background worker to keep request latency low.  
  - **Failure mode**: Golden‑set drift—periodically re‑evaluate and update the set to avoid false positives.

## Takeaway Checklist & Next Steps

- **Prompt‑design checklist** – Verify the prompt schema (JSON or plain‑text template), include few‑shot examples, enforce guardrails (stop‑words, length limits), and set a temperature that balances creativity vs determinism (e.g., `temperature=0.2` for predictable output).  
  ```python
  prompt = f"""\
  {schema}
  Example: {example}
  User: {user_input}
  Assistant:"""
  response = llm.generate(prompt, temperature=0.2)
  ```
  *Why*: Consistent prompts reduce hallucinations and simplify downstream parsing.

- **Cost‑control hooks** – Wire a token estimator before the API call, batch requests when possible, and define a fallback hierarchy (cached answer → cheaper model → static default).  
  ```python
  tokens = len(tokenizer.encode(prompt))
  if tokens > MAX_TOKENS:
      raise ValueError("Prompt too large")
  ```

- **Observability stack** – Emit structured logs (prompt, token count, latency), expose Prometheus metrics (`llm_latency_seconds`, `llm_error_total`), trace the request with OpenTelemetry, and run synthetic health checks that invoke a known prompt and assert the expected token pattern.  
  Flow: Request → Prompt → LLM → Response → Log/Metric → Trace.

- **Staged rollout plan** – Deploy to dev, promote to a canary serving 5 % of traffic, then scale to full traffic. Automate regression tests that compare hallucination rate against a baseline (e.g., `assert hallucination_rate < 0.02`).  
  *Edge case*: Canary spike in latency → roll back automatically.

- **Advanced exploration** – When the baseline is stable, prototype retrieval‑augmented generation (RAG) to inject external knowledge, experiment with RLHF‑fine‑tuned checkpoints for domain specificity, and add policy‑driven content filters that reject disallowed topics before returning a response.  
  *Trade‑off*: RAG adds latency but improves factuality; RLHF increases training cost.
