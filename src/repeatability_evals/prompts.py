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


# --------------------------------------------------------------------------------------
# Multi-dimension rubric. One LLM call scores each dimension 1-5; ``RubricJudge`` combines
# them with the weights below. Edit, drop or add dimensions by passing your own ``Dimension``s.
# --------------------------------------------------------------------------------------
from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class Dimension:
    name: str
    description: str  # what 5 vs 1 means; shown to the judge verbatim
    weight: float = 1.0


DEFAULT_DIMENSIONS = (
    Dimension(
        "core_facts",
        "Do both responses make the same central claims, conclusions or answer? "
        "5 = same core answer; 3 = same topic but the main conclusion is partly different or "
        "one omits part of it; 1 = different or contradictory core answers.",
        0.40,
    ),
    Dimension(
        "specifics",
        "Do the concrete specifics agree: names, numbers, dates, versions, identifiers, "
        "quantities, units, cited sources? "
        "5 = all shared specifics match; 3 = some differ or are missing; "
        "1 = specifics conflict. If neither response contains specifics, score 5.",
        0.25,
    ),
    Dimension(
        "supporting_detail",
        "Do the non-core details agree: explanations, examples, caveats, background, extra "
        "context? 5 = same supporting information; 3 = noticeable additions or omissions; "
        "1 = largely different supporting material. Differences here matter less than core facts.",
        0.15,
    ),
    Dimension(
        "stance",
        "Do the responses take the same stance and certainty: direct answer vs hedge vs refusal "
        "vs 'I don't know', the same recommendation or next step, similar confidence? "
        "5 = same stance; 3 = same direction but different confidence or caveats; "
        "1 = opposite stance (e.g. one answers, one declines).",
        0.10,
    ),
    Dimension(
        "format",
        "Do the responses have the same form: structure (list vs prose vs table), length, "
        "ordering, level of verbosity? 5 = essentially the same form; 3 = noticeably different "
        "structure or length; 1 = completely different form. Ignore wording that doesn't change "
        "structure.",
        0.10,
    ),
)

RUBRIC_PROMPT_TEMPLATE = """\
You are comparing two responses that were produced independently for the same input. \
Judge how closely they agree on each dimension below.

Input (may be empty):
{query}

Response A:
{item_a}

Response B:
{item_b}

Score each dimension from 1 (completely different) to 5 (identical):
{rubric}

Guidelines:
- Judge agreement between the two responses only. Do not judge which one is better or more \
correct; two equally wrong responses that say the same thing agree.
- Score each dimension independently. A formatting difference must not lower core_facts, and \
vice versa.
- Judge meaning, not surface wording.

Reply with JSON only: \
{"reason": "<one or two sentences on the main differences>", "scores": {%s}}
"""


def build_rubric_prompt(dimensions=DEFAULT_DIMENSIONS) -> str:
    rubric = "\n".join(f"- {d.name}: {d.description}" for d in dimensions)
    keys = ", ".join(f'"{d.name}": <integer 1-5>' for d in dimensions)
    return RUBRIC_PROMPT_TEMPLATE.replace("{rubric}", rubric).replace("%s", keys)
