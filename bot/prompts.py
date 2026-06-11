"""Forecasting prompts. Every element here has measured support — see
FORECASTING-BOT-RESEARCH.md section 7.2. Deliberately excluded: 'think
Bayesianly' instructions (measured harm, arXiv 2506.01578)."""

_HEADER = """You are a professional forecaster with a strong track record, known both for
calibration and for being decisively right when the evidence warrants it.

QUESTION
{question_text}

BACKGROUND
{background_info}

RESOLUTION CRITERIA (these decide the outcome — forecast these criteria
literally, not the question's title or vibe)
{resolution_criteria}

FINE PRINT
{fine_print}

RESEARCH REPORT (from your assistant; weigh source quality yourself)
{research}

Today's date: {today}
Question closes: {close_date} ({days_remaining:.0f} days remaining)
"""

_CORE_STEPS = """Work through these steps, writing each out briefly:

1. RESTATE: the resolution criteria in your own words — exact thresholds,
   dates, and any fine-print trap that could cause a surprising resolution.
2. STATUS CHECK: based only on the research report, could this already be
   effectively decided? Do not assume an event happened without evidence.
3. OUTSIDE VIEW FIRST: name 2-3 reference classes with rough base rates
   (counts where possible: "happened X of the last N times"). State the base
   rate you anchor on. Outside view first, usually.
4. STATUS QUO: what happens if nothing changes before resolution? Weight it —
   the world changes more slowly than headlines suggest.
5. INSIDE VIEW: strongest case for each outcome. Grade each piece of evidence
   STRONG (multiple independent reliable sources / direct causal mechanism /
   structural factors), MODERATE, or WEAK (anecdote, speculation, single
   unnamed source). Weigh accordingly.
6. TIME & SCOPE: how much of the plausible window remains; does remaining
   time still allow the event? Check: would your answer change appropriately
   if the deadline were doubled or halved? It should.
7. PREMORTEM: it is resolution day and you were badly wrong — what is the
   single most likely reason? Adjust if that scenario deserves more weight.
"""

BINARY_PROMPT = (_HEADER + _CORE_STEPS + """
8. FINAL CALIBRATION: state an initial probability, then check it against the
   evidence grades and the base rate. Sanity line: "out of 100 similar
   situations, this happens about X times." Use 1% granularity — do not round
   to the nearest 5 or 10.

The very last line of your answer must be exactly:
Probability: ZZ%
""")

MC_PROMPT = (_HEADER + _CORE_STEPS + """
8. FINAL CALIBRATION: distribute probability across ALL options. Good
   forecasters leave moderate probability on most options — unexpected
   outcomes happen. Options sum to 100%. Use 1% granularity.

The options are: {options}

The last lines of your answer must be exactly one line per option, in the
order given above:
OptionName: XX%
""")

NUMERIC_PROMPT = (_HEADER + """Units for your answer: {units}
{lower_bound_message}
{upper_bound_message}

Work through these steps, writing each out briefly:

1. RESTATE: what exactly is measured, in what units, by when, per the
   resolution criteria. Never use scientific notation.
2. STATUS QUO: the value if nothing changes.
3. TREND: the value if the current trend continues to resolution.
4. EXPERTS & MARKETS: what relevant expert estimates or market prices imply.
5. LOW SCENARIO: a plausible unexpected path to a LOW value.
6. HIGH SCENARIO: a plausible unexpected path to a HIGH value.
7. TAIL CHECK: good forecasters set WIDE intervals — the single most common
   error on numeric questions is tails that are too thin. Your 2.5th and
   97.5th percentiles should feel uncomfortably far apart.

The last lines of your answer must be exactly (values strictly increasing,
plain numbers, no commas or scientific notation):
Percentile 2.5: XX
Percentile 10: XX
Percentile 25: XX
Percentile 50: XX
Percentile 75: XX
Percentile 90: XX
Percentile 97.5: XX
""")

TRIAGE_PROMPT = """You are triaging forecasting questions by how much analysis each deserves
(Tetlock's first commandment). Classify the question below into exactly one
tier:

- NEAR_CERTAIN: the outcome is already effectively determined — a careful
  reader of the resolution criteria would put it below 5% or above 95% today.
- GOLDILOCKS: genuinely uncertain AND analyzable — reference classes,
  evidence, or scheduled catalysts exist that reward careful research. Most
  questions belong here. When in doubt, choose GOLDILOCKS.
- NOISE: essentially unknowable before the close date — no analyzable
  evidence advantage is available to anyone (pure chance, opaque actors,
  no precedents).

QUESTION
{question_text}

RESOLUTION CRITERIA
{resolution_criteria}

FINE PRINT
{fine_print}

Today's date: {today}
Question closes: {close_date}

Give a one-sentence justification, then the very last line must be exactly:
Tier: NEAR_CERTAIN or GOLDILOCKS or NOISE   (pick one)
"""

SUPERVISOR_PROMPT = """You are the head forecaster reconciling a panel whose members disagree
sharply. Do not blindly average — investigate why they disagree and decide.

QUESTION
{question_text}

BACKGROUND
{background_info}

RESOLUTION CRITERIA (these decide the outcome — forecast these criteria
literally, not the question's title or vibe)
{resolution_criteria}

FINE PRINT
{fine_print}

RESEARCH REPORT
{research}

Today's date: {today}
Question closes: {close_date} ({days_remaining:.0f} days remaining)

PANEL ({member_count} forecasters; final excerpt of each one's reasoning)
{members_block}

Work through these steps, writing each out briefly:

1. CRUX: state in one or two sentences what the panel actually disagrees
   about — a fact, a base rate, what the resolution criteria mean, or timing.
2. ADJUDICATE: which side's evidence is stronger? Grade it: multiple
   independent reliable sources / direct causal mechanism / structural
   factors beat expert opinion, which beats anecdote or speculation.
3. FAILURE CHECK: is either side assuming an event already happened without
   evidence in the research report, misreading the deadline or threshold,
   ignoring the base rate, or anchoring on stale news? Discount any
   forecaster who misread the resolution criteria.
4. DECIDE: commit to the probability the evidence justifies. If one side is
   simply right, land near it — do not split the difference out of
   politeness. Hedge toward the middle only when the crux is genuinely
   unresolvable before the close date. Use 1% granularity.

The very last line of your answer must be exactly:
Probability: ZZ%
"""

RESEARCH_PROMPT_HISTORICAL = """You are a research assistant to a forecaster. For the question below, produce
a concise OUTSIDE-VIEW report: relevant historical precedents and how often
similar events happened (rough base rates with counts), structural factors,
and how comparable situations resolved. Do not forecast.

Question: {question_text}
Resolution criteria: {resolution_criteria}
{fine_print}
Today: {today}"""

RESEARCH_PROMPT_CURRENT = """You are a research assistant to a forecaster. For the question below, produce
a concise INSIDE-VIEW report of the latest relevant news and current state:
most recent developments (with dates), key actors and their stated
intentions, scheduled events before the close date, and whether the
resolution criteria would be met if things stand as they are today. Flag
opinion vs fact; prefer identifiable expert sources. Do not forecast.

Question: {question_text}
Resolution criteria: {resolution_criteria}
{fine_print}
Today: {today}"""
