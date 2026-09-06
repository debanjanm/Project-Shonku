---
name: sec-filing-periods
description: How to search across SEC 10-Q filings when the user names or compares specific fiscal quarters/years (e.g. "Q1 2026 vs Q2 2026", "how did revenue change from last quarter"). Use whenever a question involves a specific filing period or a comparison between periods.
---

# SEC filing period-exact search

`search_knowledge_base` runs a hybrid search that, when the query names an
exact year AND quarter (e.g. "2026", "Q1"), automatically narrows results
to chunks from that exact filing — chunks from other quarters of the same
company are excluded from ranking entirely, not just down-weighted. This
exists specifically because dense embeddings alone can't reliably tell
apart two quarters of the same company's near-identical boilerplate.

**Consequence for how you should search:**

- A query naming one exact period ("Apple Q1 2026 revenue") already gets
  this narrowing for free — just search normally.
- A query comparing **two or more periods** ("how did Apple's revenue
  change from Q1 to Q2 2026?") must NOT be sent as a single blended query —
  the period filter can only lock onto one year+quarter pair per search, so
  a blended query either matches neither period cleanly or arbitrarily
  favors whichever period-mention appears first in the text. Instead, run
  **one `search_knowledge_base` call per period**, each with that period's
  year+quarter spelled out explicitly, then compare the two answers
  yourself in your response.
- A query with no exact period ("what are Apple's main risk factors")
  intentionally gets no narrowing — search normally, results may span
  multiple filings, which is correct for that kind of question.

Example — user asks "How much did Apple's revenue grow from Q1 to Q2
2026?":
1. `search_knowledge_base("Apple Q1 2026 revenue")`
2. `search_knowledge_base("Apple Q2 2026 revenue")`
3. Compute the difference yourself from the two cited figures, cite both
   sources.
