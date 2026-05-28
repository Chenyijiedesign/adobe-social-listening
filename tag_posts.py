"""
Reddit Post Tagger
Tags posts with: Emotion, Creative Activity, AI Related, AI relation, Adobe Related
Processes untagged rows in batches via Claude API.
"""

import csv
import json
import os
import time
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(_DIR, "Reddit_Social Listening_Project Adobe - combined.csv")
OUTPUT_FILE = os.path.join(_DIR, "Reddit_Social Listening_Project Adobe - combined_tagged.csv")
CHECKPOINT_FILE = os.path.join(_DIR, "tagging_checkpoint.json")
BATCH_SIZE = 25

TAG_COLS = ["Emotion", "Creative Activity", "AI Related", "AI relation", "Adobe Related"]

SYSTEM_PROMPT = """You are a social media research tagger. Your job is to classify Reddit posts for a study on creative communities.
Return ONLY valid JSON — no explanations, no markdown, no code blocks."""

TAGGING_PROMPT = """Tag each Reddit post below. Focus on TITLE and BODY first; use topComments only when title/body are ambiguous.

=== TAG DEFINITIONS ===

EMOTION (pick exactly ONE):
- excitement: optimism, wonder, enthusiasm about something positive
- curiosity: exploratory learning, asking questions to understand
- pride: sharing accomplishment or mastery
- liberation: empowerment, newfound access or freedom
- anxiety: fear, worry, uncertainty
- fatigue: burnout, overload, exhaustion
- skepticism: distrust, critical evaluation, doubt
- resentment: anger, frustration, complaint
- grief: loss, mourning, sadness
- nostalgia: longing for the past, backward-looking
- humor: memes, irony, jokes, playful tone
- neutral: low emotional signal, informational

CREATIVE ACTIVITY (pick exactly ONE):
- showcase: sharing finished or in-progress creative work
- workflow_share: explaining how you did something (process-focused)
- help_request: asking for guidance, troubleshooting
- discussion: conversation, debate, opinion sharing
- emotional_expression: venting feelings, personal reflection
- meme: humor-first posts, image macros
- experimentation: exploratory testing of tools or techniques
- critique: evaluating/reviewing work or tools
- tutorial: teaching, step-by-step instructions
- prediction: future speculation

AI RELATED (TRUE or FALSE):
- TRUE if AI (generative AI, LLMs, image AI, ChatGPT, Midjourney, Stable Diffusion, Firefly, Sora, Gemini, Claude, etc.) is directly mentioned or is central to the post
- FALSE if no AI involvement

AI RELATION (pick exactly ONE):
- none: use when AI Related is FALSE, OR when AI is mentioned in the post but there is no clear signal about HOW the person relates to or uses AI
- experimentation: playfully exploring AI capabilities
- augmentation: AI assists an existing human workflow
- collaboration: actively co-creating something with AI
- orchestration: directing or chaining multiple AI systems
- dependency: heavily reliant on AI to function
- optimization: using AI primarily for speed/productivity gains
- adaptation: pragmatically learning to integrate AI into practice
- reluctant_adoption: using AI despite dissatisfaction or resistance
- rejection: explicitly refusing or resisting AI
- replacement: AI is substituting/displacing human work
- identity_shift: AI is changing who someone is as a professional

IMPORTANT: "none" is valid even when AI Related is TRUE — use it when AI is mentioned but the person's relationship/behavior toward AI is unclear or not the focus of the post.

ADOBE RELATED (TRUE or FALSE):
- TRUE if the post is specifically about Adobe products (Photoshop, Illustrator, Premiere Pro, After Effects, InDesign, Lightroom, Acrobat, Firefly, Adobe Stock, Creative Cloud, etc.) or the Adobe company
- FALSE otherwise

=== OUTPUT FORMAT ===
Return a JSON array. Each object must have exactly these keys:
{
  "id": "<post id>",
  "emotion": "<label>",
  "creative_activity": "<label>",
  "ai_related": "TRUE or FALSE",
  "ai_relation": "<label>",
  "adobe_related": "TRUE or FALSE"
}

=== POSTS TO TAG ===
"""

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {}

def save_checkpoint(tagged_ids):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(tagged_ids, f)

def tag_batch(client, batch, retries=3):
    posts_text = ""
    for i, row in enumerate(batch, 1):
        body = (row.get("body") or "")[:400].strip()
        comments = (row.get("topComments") or "")[:200].strip()
        posts_text += f"""
--- Post {i} ---
ID: {row['id']}
Subreddit: r/{row.get('subreddit', '')}
Title: {row.get('title', '')}
Body: {body}
TopComments: {comments}
"""

    prompt = TAGGING_PROMPT + posts_text + "\nReturn only the JSON array:"

    for attempt in range(retries):
        try:
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = message.content[0].text.strip()

            # Strip markdown code fences if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1]).strip()

            results = json.loads(response_text)
            return results

        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))

    return None

def is_tagged(row):
    """A row is considered human-tagged if Emotion is filled (primary indicator)."""
    return bool(row.get("Emotion", "").strip())

def needs_adobe_fill(row):
    """Row is human-tagged but Adobe Related was left blank."""
    return is_tagged(row) and not row.get("Adobe Related", "").strip()

ADOBE_KEYWORDS = [
    "adobe", "photoshop", "illustrator", "premiere", "after effects",
    "indesign", "lightroom", "acrobat", "firefly", "creative cloud",
    "adobe stock", "substance", "fresco", "aero", "rush", "xd",
    "adobe cc", "adobe ai"
]

def infer_adobe_related(row):
    """Rule-based Adobe Related for partially-tagged rows."""
    text = f"{row.get('title','')} {row.get('body','')}".lower()
    if any(kw in text for kw in ADOBE_KEYWORDS):
        return "TRUE"
    if row.get("source_file", "") == "adobe.json":
        return "TRUE"
    return "FALSE"

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Run: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic library not installed. Run: pip3 install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load CSV
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Total rows: {len(rows)}")

    # Load checkpoint
    checkpoint = load_checkpoint()
    print(f"Checkpoint: {len(checkpoint)} rows already tagged in previous runs")

    # Handle partially-tagged rows (human-tagged but Adobe Related is blank)
    partial_fixed = 0
    for row in rows:
        if needs_adobe_fill(row):
            row["Adobe Related"] = infer_adobe_related(row)
            partial_fixed += 1
    if partial_fixed:
        print(f"Fixed {partial_fixed} partially-tagged rows (Adobe Related was blank)")

    # Separate fully untagged rows
    to_tag = []
    for row in rows:
        if is_tagged(row):
            continue  # Human-tagged — skip
        if row["id"] in checkpoint:
            # Apply checkpoint tags from previous run
            tags = checkpoint[row["id"]]
            row["Emotion"] = tags.get("emotion", "")
            row["Creative Activity"] = tags.get("creative_activity", "")
            row["AI Related"] = tags.get("ai_related", "FALSE")
            row["AI relation"] = tags.get("ai_relation", "none")
            row["Adobe Related"] = tags.get("adobe_related", "FALSE")
        else:
            to_tag.append(row)

    print(f"Rows needing full tags: {len(to_tag)}")

    # Process in batches
    total_batches = (len(to_tag) + BATCH_SIZE - 1) // BATCH_SIZE
    tagged_count = 0

    for batch_num in range(total_batches):
        batch = to_tag[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE]
        print(f"\nBatch {batch_num + 1}/{total_batches} ({len(batch)} posts)...", end=" ", flush=True)

        results = tag_batch(client, batch)

        if results is None:
            print("FAILED — skipping batch, will retry next run")
            continue

        # Map results back by ID
        results_by_id = {r["id"]: r for r in results}

        for row in batch:
            rid = row["id"]
            if rid in results_by_id:
                tags = results_by_id[rid]
                row["Emotion"] = tags.get("emotion", "neutral")
                row["Creative Activity"] = tags.get("creative_activity", "discussion")
                row["AI Related"] = str(tags.get("ai_related", "FALSE")).upper()
                row["AI relation"] = tags.get("ai_relation", "none")
                row["Adobe Related"] = str(tags.get("adobe_related", "FALSE")).upper()
                checkpoint[rid] = tags
                tagged_count += 1
            else:
                print(f"  Warning: no result for {rid}")

        save_checkpoint(checkpoint)
        print(f"done ✓  (total tagged this run: {tagged_count})")

        # Polite rate limit pause
        if batch_num < total_batches - 1:
            time.sleep(0.5)

    # Write output CSV (preserve original column order)
    print(f"\nWriting {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    all_tagged = sum(1 for r in rows if is_tagged(r) or r["id"] in checkpoint)
    print(f"\n✅ Done!")
    print(f"   Total rows: {len(rows)}")
    print(f"   Tagged rows: {all_tagged}")
    print(f"   Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
