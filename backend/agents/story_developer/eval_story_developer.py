"""Story Developer eval harness. No ground truth exists (unlike docqa's
ragbench benchmark or recommendation's leave-one-out design) — this is pure
generative writing, nothing to check recall against. LLM-as-judge instead:
run a fixed set of varied seed ideas through the real agent, have a second
LLM score the output against fixed, checkable criteria drawn directly from
what the agent's own system prompt + screenplay-formatting skill already
promise (title/genre, logline, three-act synopsis, 3-4 characters, a
properly formatted scene, internal coherence). Not a perfect signal — a
judge model has no more ground truth than the agent does — but scoring
specific criteria is a real signal, not vibes.

Run:
    python -m backend.agents.story_developer.eval_story_developer
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(ROOT / ".env")

from openai import OpenAI  # noqa: E402

from backend.agents.story_developer.agent import make_story_agent  # noqa: E402
from backend.config import get_router_config  # noqa: E402
from backend.logging_config import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

REPORT_PATH = ROOT / "data" / "raw" / "story_developer_eval_report.json"

SEED_IDEAS = [
    "A detective who can't trust her own memories investigates a murder in a floating city.",
    "Two rival food-truck owners fall for each other during a city-wide cooking competition.",
    "A retired heist crew is pulled back for one last job to save one of their own.",
    "A shy teenager discovers their late grandmother's diary describes a war that never happened.",
    "A washed-up astronaut is the only one who believes a signal from Mars is real.",
    "A small-town sheriff investigates a string of disappearances tied to an old family curse.",
    "Two estranged siblings inherit a haunted vineyard and must run it together for a year.",
    "A hitman decides to protect his next target instead of killing her.",
    "A historian time-slips into 1920s Paris and can't find a way back.",
    "A group of strangers wake up on a train with no memory of boarding it.",
]

JUDGE_SYSTEM_PROMPT = """You are a strict script-development editor. Score a generated story concept + scene
against fixed criteria. Always respond with valid JSON only. No markdown, no explanation outside the JSON."""

JUDGE_USER_TEMPLATE = """Evaluate this story development output against each criterion below. For each, answer
true or false.

- has_title_and_genre: has a clear title and a specific genre or genre mix.
- has_logline: has a one-sentence hook that captures the premise.
- has_three_act_synopsis: has a synopsis with a recognizable setup, confrontation, and resolution.
- has_3_to_4_characters: profiles 3-4 named characters with distinct motivations.
- scene_has_slugline: the scene opens with a standard slugline (e.g. "INT. LOCATION - NIGHT").
- scene_has_dialogue: the scene contains actual character dialogue, not just description.
- coherent: the scene and characters are consistent with the synopsis, no contradictions.

Return a JSON object with exactly these 7 boolean keys.

OUTPUT TO EVALUATE:
{output}"""


def _judge(client: OpenAI, model: str, output: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_TEMPLATE.format(output=output)},
            ],
            temperature=0.0,
            max_tokens=300,
            timeout=30,
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception:
        logger.exception("judge call failed")
        return {}


CRITERIA = (
    "has_title_and_genre",
    "has_logline",
    "has_three_act_synopsis",
    "has_3_to_4_characters",
    "scene_has_slugline",
    "scene_has_dialogue",
    "coherent",
)


def run_eval(seed_ideas: list[str] = SEED_IDEAS) -> None:
    configure_logging(plain=True)
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
    judge_model = get_router_config().hard_model

    hits = {c: 0 for c in CRITERIA}
    overall_hits = 0
    samples = []

    for i, idea in enumerate(seed_ideas, 1):
        logger.info("[%d/%d] %r", i, len(seed_ideas), idea)
        agent = make_story_agent()
        result = agent.invoke({"messages": [{"role": "user", "content": idea}]})
        output = result["messages"][-1].content

        scores = _judge(client, judge_model, output)
        passed = {c: bool(scores.get(c, False)) for c in CRITERIA}
        for c in CRITERIA:
            if passed[c]:
                hits[c] += 1
        all_pass = all(passed.values())
        if all_pass:
            overall_hits += 1
        samples.append({"idea": idea, "scores": passed, "all_pass": all_pass})
        logger.info("  -> %s", passed)

    n = len(seed_ideas)
    report = {
        "sample_size": n,
        "criteria_pass_rate": {c: round(hits[c] / n, 4) for c in CRITERIA},
        "overall_pass_rate": round(overall_hits / n, 4),
        "samples": samples,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    logger.info("=== Story Developer eval (n=%d) ===", n)
    for c in CRITERIA:
        logger.info("%-24s %.1f%%", c, report["criteria_pass_rate"][c] * 100)
    logger.info("Overall pass rate: %.1f%%", report["overall_pass_rate"] * 100)
    logger.info("report written to %s", REPORT_PATH)


def main() -> None:
    run_eval()


if __name__ == "__main__":
    main()
