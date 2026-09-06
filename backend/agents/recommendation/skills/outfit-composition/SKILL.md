---
name: outfit-composition
description: How to handle a request for a full outfit or a look built from multiple distinct items (e.g. "an outfit for a summer wedding", "put together a casual weekend look"), rather than a single product.
---

# Composing a multi-item outfit

`search_products` returns items matching one query — it cannot represent
"a top AND a bottom AND shoes" in a single call, since the underlying CLIP
embedding + vector search is built around one coherent product concept per
query. Sending a compound request as one query (e.g. "outfit for a summer
wedding: dress, shoes, bag") will bias results toward whichever single item
type dominates the phrasing, not return a balanced set.

**Instead, decompose the request into one `search_products` call per item
category implied by the ask**, using the occasion/style words from the
user's request in each:

- Identify which garment/accessory categories the outfit needs (e.g. top,
  bottom, shoes, and — only if clearly implied — a bag or accessory)
- Run one `search_products` call per category, each carrying the shared
  style/occasion context (e.g. "summer wedding guest dress",
  "summer wedding guest heels")
- Present the results together as one cohesive outfit in your final answer,
  organized by category, not as separate unrelated searches

Example — user asks "put together a casual weekend outfit for a man":
1. `search_products("men's casual weekend top")`
2. `search_products("men's casual weekend bottoms")`
3. `search_products("men's casual weekend shoes")`
4. Present all three as one outfit, grouped by category, each with its own
   match reason and `[Image: ...]` tag preserved.

Don't over-decompose a request that's already about one item — a plain
"blue running shoes" query is a single `search_products` call, not an
excuse to search for a whole outfit around it.
