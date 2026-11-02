"""Default prompt for the LLM judge. Placeholders: {query}, {item_a}, {item_b}."""

SEMANTIC_EQUIVALENCE_PROMPT = """\
You are comparing two responses that were produced independently for the same input. \
Decide how closely they agree in meaning.

Input (may be empty):
{query}

Response A:
{item_a}

Response B:
{item_b}

Rate their semantic equivalence on a 1-5 scale:
5 = Identical meaning: same core facts, no meaningful differences
4 = Very similar: only minor phrasing or presentational differences
3 = Partially equivalent: some differences in detail or completeness
2 = Mostly different: notable factual or informational differences
1 = Completely different information: little to no overlap

Guidelines:
- Judge meaning, not wording, length, formatting, or the order of points.
- Do not judge which response is better or more correct. Only judge whether they agree; \
two equally wrong responses that say the same thing score 5.
- Extra detail in one response lowers the score only when it adds or changes substantive \
information that the other lacks or contradicts.

Reply with JSON only: {"reason": "<one or two sentences>", "score": <integer 1-5>}
"""
