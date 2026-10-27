# Evaluating the SEC 10-K Agent

A learning-oriented walkthrough of the eval harness — what it measures, why
those metrics, how the formulae work, and how to read a report. Read top to
bottom; each section builds on the previous.

---

## 1. Why we evaluate at all

The agent's product promise is **"every claim is grounded in a real SEC
filing; if it can't be grounded, the agent refuses."** That is a *behavioral*
property of the system as a whole. Unit tests cannot prove it:

- A unit test of `cite_check` shows the *gate* works in isolation.
- It cannot show the gate is even *reached* on a real question.
- It cannot show that the LLM, given a real chunk, paraphrases instead of
  hallucinating.
- It cannot show that the agent refuses a question Perplexity would
  confidently lie about.

Eval harnesses fill that gap. We pick a fixed set of representative
questions, run the *whole* agent against them, and measure outcomes.

This is also a **Definition-of-Done blocker** in CLAUDE.md: ≥30 golden Q/A,
≥85% citation accuracy, <5% hallucination rate, wired into CI as a failing
gate. Without the harness we cannot ship.

### Business framing

| Stakeholder | What they care about | Which metric proves it |
| --- | --- | --- |
| End user | "Can I trust the answer?" | hallucination_rate |
| End user | "Did it actually answer my question?" | false_refusal_rate |
| Compliance / legal | "Will it lie about a public filing?" | hallucination_rate, refusal_accuracy |
| PM vs Perplexity | "Does it refuse where Perplexity bluffs?" | refusal_accuracy on adversarial |
| Engineer on call | "Did my last PR regress anything?" | all four, vs the previous run |
| Investor / demo | "Does this work?" | report.md, one screenshot |

---

## 2. The four metrics — definitions and formulae

Let `R` be the set of all rows we ran. Each row carries an `Expected` (what
the dataset says should happen) and an outcome derived from the agent's
final state.

Define disjoint subsets of `R`:

- `G` = **golden** rows (`expected.should_refuse == False`).
- `A` = **adversarial** rows (`expected.should_refuse == True`).
- `G_answered` = `{r ∈ G : not refused}` — golden rows the agent actually
  answered.
- `G_refused`  = `{r ∈ G : refused}`     — golden rows the agent refused
  (these are the false-refusals).
- `A_refused`  = `{r ∈ A : refused}`     — adversarial rows correctly
  refused.

### 2.1 citation_accuracy

> Of the golden rows the agent answered, what fraction cited the *expected*
> filing (and Item section, when applicable)?

```
citation_accuracy = |{ r ∈ G_answered : citation_correct(r) }| / |G_answered|
```

`citation_correct(r)` is true iff:
- at least one claim's `source_ref.accession_no` (after dash-stripping)
  matches one of `expected.accession`, AND
- if `expected.pipeline == "prose"` and `expected.item_section` is set,
  some cited source has that `item`.

Implemented in `metrics.score_citation` (`metrics.py:129`).

**Vacuous-pass rule.** If `|G_answered| = 0` (e.g. running a smoke set of
five adversarial rows), we return **1.0** rather than `0/0`. This avoids a
fake fail when the metric does not apply. (See `_accuracy`,
`metrics.py:298`.)

**Target:** ≥ 0.85.

### 2.2 hallucination_rate

> Of the rows where we *can* judge truthfulness, what fraction contained an
> unsupported claim?

A row is "judgeable" if it produced a numeric answer (then we compare
against the XBRL fact) **or** the LLM-as-judge scored it (semantic check).
A row is "hallucinated" if either signal fires.

```
H_eligible = { r ∈ G_answered : numeric_correct ∈ {T,F} ∨ judge_supported ∈ {T,F} }
H_wrong    = { r ∈ H_eligible : numeric_correct = F ∨ judge_supported = F }

hallucination_rate = |H_wrong| / |H_eligible|
```

Two layers because they catch different failure modes:

1. **Structural (numeric).** The agent says "$391 billion"; XBRL says
   `391_035_000_000`. Decimal compared with tolerance (default 0.5%). Catches
   wrong digits, magnitude errors, and rounding that crosses the tolerance.
2. **Semantic (judge).** At runtime, cite_check enforces a *structural*
   prose check: the LLM emits a verbatim `evidence_quote` alongside its
   paraphrased `claim.text`, and cite_check substring-matches the quote
   (whitespace-normalized) against the chunk text — guaranteeing the
   source contains some literal text supporting the claim. The judge
   complements that at eval time: it reads the full `(question,
   answer, sources)` and asks whether the *answer as a whole* is
   supported — catching cases where the quote is real but used out of
   context, or where the answer extrapolates beyond what the quote
   says. Structured output, cached on disk by
   `sha256(model + question + answer + sources)`.

Implemented in `aggregate` (`metrics.py:306`) and `judge.py`.

**Target:** < 0.05. *Strictly less than* — equality fails the gate. This is
deliberate: hallucination is the differentiator, the bar is set high.

### 2.3 refusal_accuracy

> Of the adversarial rows (designed to be unanswerable from filings), what
> fraction did the agent correctly refuse?

```
refusal_accuracy = |A_refused| / |A|
```

Vacuous-pass to 1.0 when `|A| = 0`.

**Target:** = 1.00. This is the differentiator vs Perplexity. Every
adversarial row is something a generic LLM will confidently make up. We
must refuse all of them.

### 2.4 false_refusal_rate

> Of the golden rows (which *should* be answerable), what fraction did the
> agent unnecessarily refuse?

```
false_refusal_rate = |G_refused| / |G|
```

Plain `0/0 → 0.0` (no vacuous pass — a zero-row golden set genuinely has no
false refusals).

**Target:** ≤ 0.05.

False-refusal is the failure mode of an over-cautious gate. A 100%
hallucination-free agent that refuses every question scores perfectly on
the first three metrics and is useless. This metric is the counterweight.

### 2.5 The four together

Each metric controls one failure mode; together they tile the whole space:

| Failure mode | Caught by |
| --- | --- |
| Cites the wrong filing | citation_accuracy |
| Cites the right filing but lies about its contents | hallucination_rate |
| Bluffs on a question with no source | refusal_accuracy |
| Refuses a question it could have answered | false_refusal_rate |

You cannot game one without paying in another. That is the design.

---

## 3. The datasets

### 3.1 `golden_dataset.json` — what should be answered

~32 rows, each with:

```jsonc
{
  "id": "g_aapl_revenue_fy24",
  "question": "What was Apple's total revenue in fiscal year 2024?",
  "pipeline": "xbrl",            // xbrl | prose | cross | refuse
  "ticker": "AAPL",
  "fiscal_year": 2024,
  "accession": "0000320193-24-000123",
  "concept_key": "revenue",
  "value": "391035000000",       // Decimal as string
  "tolerance_pct": 0.5,
  "tags": ["numeric", "revenue"]
}
```

Pipelines:

- **xbrl** — numeric question. Scored on numeric tolerance + accession.
- **prose** — qualitative question. Scored on accession + Item section
  (and optional keyword coverage).
- **cross** — both numeric and prose claims expected.
- **refuse** — golden rows can also be expected refusals (rare; usually
  adversarial covers this).

### 3.2 `adversarial.json` — what should be refused

10 rows, each with `should_refuse: true` and a `refusal_kinds` tag. The
tags map to the agent's actual refusal taxonomy:

| `refusal_kinds` | Meaning | Example |
| --- | --- | --- |
| `out-of-source` | Answerable, but not from a 10-K | "What's AAPL's stock price right now?" |
| `out-of-scope` | Outside our 50-company / post-2015 / US-only universe | "IBM 2010 10-K", "Toyota annual report" |
| `not-in-filing` | Looks like a 10-K question, but the fact does not exist | "Apple's customer satisfaction score" |

These are deliberately the questions Perplexity *will* answer — confidently
and wrongly. Our refusal of them is the demo.

### 3.3 Fields that matter for scoring

- `accession`: dash-tolerant. Both `0000320193-24-000123` and
  `0000320193241000123` match.
- `tolerance_pct`: percentage difference allowed on `value`. Default 0.5%
  absorbs USD-vs-USD rounding without absorbing magnitude bugs.
- `keywords`: optional hint for prose answers. Coverage = fraction present
  in the answer; not a gate, just a signal in the report.

### 3.4 Keeping the dataset in sync with the DB — `audit_dataset.py`

A dataset row is a *claim about reality* — "AAPL's FY2024 revenue is
391,035M, found in accession 0000320193-24-000123, in this filing".
Reality is the ingested DB. When the two drift (dataset author copied
a wrong year's figure, ticker isn't yet ingested, EDGAR's fiscal-year
tagging is offset), every eval run flags the row as a regression even
though the agent is right.

`evals/audit_dataset.py` reconciles the dataset against the DB:

- Drops rows whose ticker isn't in `companies` (out-of-universe).
- Replaces `accession` with the actual ingested filing for
  `(cik, fiscal_year)`.
- Replaces `value` with the actual XBRL fact, looked up the same way
  the agent does at runtime (`concept` = friendly key, ordered by
  `period_end DESC`).
- Drops rows whose `(cik, concept_key, fiscal_year)` does not resolve
  to any XBRL fact (typically EDGAR's quirky fiscal-year tagging —
  e.g. GOOGL's calendar-2024 revenue is stored under `fiscal_year=2025`
  because they filed in 2025; track separately as an agent-side bug).

Run `cd backend && uv run python ../evals/audit_dataset.py` for a
dry-run; add `--apply` to overwrite `golden_dataset.json` (the previous
file is saved to `golden_dataset.json.bak`). Re-run after every
re-ingestion of a filing.

---

## 4. Per-row scoring (what `score_row` actually does)

Defined at `metrics.py:220`. Pseudocode:

```python
refused = (state.question_type == "REFUSE") or bool(state.refusal_reason_final)

if expected.should_refuse:
    out.refused_correctly = refused
    return out                          # nothing else applies

out.false_refusal = refused
if refused:
    return out                          # citation/numeric moot

out.citation_correct = score_citation(expected, state)

if pipeline in {"xbrl", "cross"} and expected.value is not None:
    out.numeric_correct = score_numeric(expected, state)

if pipeline in {"prose", "cross"} and expected.keywords:
    out.keyword_coverage = score_keywords(expected, state)
```

A `RowOutcome` field is `None` when the metric does not apply to that row.
Aggregation simply skips `None` — that is how the four metrics ignore
irrelevant rows without bookkeeping.

### Numeric tolerance, exactly

```
diff_pct = |fact_val - expected_val| / |expected_val| * 100
ok = diff_pct ≤ tolerance_pct
```

`expected_val == 0` is special-cased to exact equality.

Example: expected `391_035_000_000`, agent surfaces `391_000_000_000`,
tolerance 0.5%. `diff_pct = 35_000_000 / 391_035_000_000 * 100 ≈ 0.00895%`
→ **pass**.

### Citation matching, exactly

1. Strip dashes from every accession.
2. Build the set of cited accessions from claim `source_ref`s; fall back to
   `state.filing.accession_no` if no claim carries one.
3. Require non-empty intersection with the expected set.
4. For prose, also require expected `item_section` to appear in the cited
   sources' `item` set.

---

## 5. The LLM-as-judge (semantic hallucination)

Substring matching cannot tell a paraphrase from a fabrication. The judge
fills that gap.

```
input  = (question, answer, [sources rendered as XBRL fact or chunk slice])
output = JudgeVerdict { supported: bool, unsupported_claims: [str], reasoning: str }
model  = gpt-4o-mini (cheap, structured output, uses OPENAI_API_KEY
                      which the rest of the stack already requires for
                      embeddings; override with JUDGE_MODEL env)
cache_key = sha256(model || question || answer || sources_json)
cache_path = evals/results/.judge_cache.json
```

Why a cache: rerunning the harness on the same dataset shouldn't burn
tokens. The key includes the model name so a model bump invalidates the
cache automatically.

Why Haiku: judging "did the answer say something the source did not"
is genuinely easier than answering the question. Frontier model unnecessary.

The judge is **not** the source of truth — it's a second opinion layered on
top of the structural numeric check. We trust the structural check
absolutely; we trust the judge enough to flag rows for human review.

---

## 6. The runner

`run_evals.py` is the wiring. Conceptually:

```python
for row in dataset:
    state = AgentState(question=row.question, request_id=...)
    t0 = perf_counter()
    final = await GRAPH.ainvoke(state)        # runs the real agent
    latency_ms = (perf_counter() - t0) * 1000
    outcome = score_row(row, final, latency_ms=latency_ms)
    if not outcome.refused and judge_enabled:
        verdict = await judge_hallucination(row.question, final)
        outcome.judge_supported = verdict.supported
        outcome.judge_unsupported_claims = verdict.unsupported_claims
    outcomes.append(outcome)

metrics = aggregate(outcomes)
report  = format_report(metrics, outcomes, thresholds)
write(evals/results/<ts>.{json,md})
sys.exit(0 if metrics.passes(thresholds) else 1)
```

Key choices:

- **In-process `GRAPH.ainvoke`** — not over HTTP. Keeps Langfuse traces
  clean, removes deploy coupling, makes failures debuggable in a debugger.
- **Non-zero exit on threshold breach** — that's the CI gate.
- **Persisted JSON + Markdown reports** under `evals/results/<timestamp>`.
  JSON for diffing, MD for humans.

CLI flags worth knowing:

| Flag | What it does |
| --- | --- |
| `--golden PATH` | Override golden dataset |
| `--adversarial PATH` | Override adversarial dataset |
| `--limit N` | Run only first N rows of each. Smoke-test mode. |
| `--no-judge` | Skip the LLM judge (free, faster, structural-only) |
| `--judge-model NAME` | Override `gpt-4o-mini` |
| `--threshold-citation X` | Loosen/tighten any threshold for an experiment |

---

## 7. Reading a report

`format_report` (`metrics.py:353`) emits something like:

```
# Eval Report

- rows:                33 (32 golden, 1 adversarial)
- citation_accuracy:   100.0%   [target ≥  85.0%] PASS
- hallucination_rate:   91.0%   [target <   5.0%] FAIL
- refusal_accuracy:    100.0%   [target ≥ 100.0%] PASS
- false_refusal_rate:   65.6%   [target ≤   5.0%] FAIL
- median_latency_ms:    3120

## Failures
- **g_msft_supplier_risk** (prose): unexpected refusal: I couldn't verify all claims...
- **g_googl_net_income_fy24** (xbrl): numeric: got 100118000000 (accn=...), expected 59972000000 ±0.5%
- ...
```

How to read it:

1. **Did each metric pass its threshold?** The four PASS/FAIL marks are the
   gate — those are what CI checks.
2. **Where did things fail?** The `## Failures` list carries one bullet per
   row whose `notes` are non-empty. The note format reveals the failure
   mode immediately:
   - `unexpected refusal: <reason>` → false_refusal. Look at the agent's
     refusal reason; usually points at cite_check or fetch.
   - `numeric: got X (accn=Y), expected Z` → either bad XBRL data or wrong
     dataset value. Check the source-of-truth before "fixing" the agent.
   - `citation: cited {…}, expected one of {…}` → wrong filing retrieved.
     Could be a Locate/Fetch bug or a stale dataset accession.
   - `item mismatch` → right filing, wrong section. Retrieval issue.
3. **Did the rate of any failure drop or climb vs the last run?** Diff
   `evals/results/<latest>.json` against the previous run.

---

## 8. Worked examples

### 8.1 A golden XBRL row passes

```
Expected: g_aapl_revenue_fy24, pipeline=xbrl, accession=0000320193-24-000123,
          value=391035000000, tolerance_pct=0.5
Agent:    refused=False, xbrl_fact.value=391035000000, accession_no=0000320193-24-000123
```

- `is_refused` → False
- `false_refusal` → False
- `score_citation`: cited accession (normalized: `0000320193241000123`)
  intersects expected set. **citation_correct = True**
- `score_numeric`: `|391035000000 - 391035000000| / 391035000000 * 100 = 0%
  ≤ 0.5%`. **numeric_correct = True**
- Judge: supported = True (number matches and answer cites the fact)
- Contributes: +1 cite_correct, +1 cite_eligible, +0 halluc_wrong,
  +1 halluc_eligible.

### 8.2 An adversarial row passes

```
Expected: a_aapl_stock_price_now, should_refuse=True
Agent:    question_type="REFUSE", refusal_reason_final="Stock prices are not in 10-K filings."
```

- `is_refused` → True
- `refused_correctly` = True
- Contributes: +1 refused_ok, +1 refusal_eligible. Nothing else.

### 8.3 A false refusal

```
Expected: g_aapl_supplier_risk, pipeline=prose, accession=0000320193-24-000123,
          item_section=1A
Agent:    refused=True, refusal_reason_final="couldn't verify all claims..."
```

- `false_refusal` = True
- Note: `unexpected refusal: couldn't verify all claims...`
- citation_correct stays `None` (we only score citation on answered rows)
- This counts in the false_refusal denominator.

### 8.4 A semantic hallucination caught only by the judge

```
Expected: g_aapl_supplier_risk, pipeline=prose
Agent:    answered with "Apple makes most of its products in Vietnam."
          (chunk actually says "limited number of suppliers in Asia",
          no mention of Vietnam)
```

- `score_citation` → True (right filing, right Item)
- No numeric claim, so `numeric_correct = None`
- Judge → `supported=False`, `unsupported_claims=["Vietnam"]`
- Contributes: +1 cite_correct, +1 halluc_wrong (semantic).

---

## 9. Common gotchas

- **Don't compare accessions as strings.** Always pass through
  `_normalize_accn`. `0000320193-24-000123` and `0000320193241000123` are
  the same filing.
- **Vacuous-pass is on accuracies, not rates.** `_accuracy` returns 1.0 on
  empty denominator; `_safe_div` returns 0.0. Hallucination-rate-empty
  legitimately *is* zero hallucinations; citation-accuracy-empty is *no
  data*, which we choose not to penalize.
- **Empty XBRL value vs zero.** `expected_val == 0` uses exact equality. If
  you ever encode an expected zero and the agent returns 1, percentage
  arithmetic would hide the failure (`|1-0|/|0|` is undefined). Special case.
- **The judge sees the answer, not the reasoning.** If the agent's draft
  contains a slip but the cited source is correct, the judge will catch it
  only if the slip is in the *answer text*. Internal scratchpads are not
  judged.
- **Latency is wall-clock, in-process.** Useful for trend, not absolute SLO
  — production has network hops the runner does not.

---

## 10. How this gates CI (target state)

- PR build: `python -m evals.run_evals --limit 10` runs a smoke subset.
  Cheap, must pass thresholds.
- `main` push: full dataset, `--no-judge` off. Stricter thresholds. Failure
  blocks deploy.
- Reports persisted as build artifacts under `evals/results/<sha>/`.
- Trend dashboard (Langfuse + the JSON output) shows the four metrics over
  time — that is the regression signal.

(#6 on the open task list — not yet wired.)

---

## 11. The mental model in one paragraph

The agent is a system that turns a question into either an answer or a
refusal. The eval harness is a function that turns *(dataset, agent
behavior)* into four numbers. Each number measures one of the four ways the
agent can fail. As long as those numbers stay inside their thresholds, the
agent is meeting its product promise. As soon as any number drifts, the
report tells you exactly which rows are responsible. That is the entire
loop: write a question, run the harness, fix the row, re-run.
