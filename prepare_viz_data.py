#!/usr/bin/env python3
"""
Prepare visualization data from the tagged Reddit CSV.
Computes TF-IDF + UMAP for the spatial scatter plot.
Output: viz_data.json  →  loaded by index.html
"""

import csv, json, re, sys
import numpy as np
from collections import Counter

INPUT  = "Reddit_Social Listening_Project Adobe - combined_tagged.csv"
OUTPUT = "viz_data.json"

# ── Tag normalization (fix non-standard values from LLM) ─────────────────────

EMOTION_NORM = {
    "frustration":          "resentment",
    "resilience":           "neutral",
    "celebration":          "excitement",
    "critique":             "skepticism",
    "emotional_expression": "neutral",
    "experimentation":      "curiosity",
    "discussion":           "neutral",
    "relief":               "neutral",
}
AI_REL_NORM = {
    "comparison":  "none",
    "frustration": "none",
}

SOURCE_CATEGORY = {
    "3d-gaming-technicalart.json": "3D & Gaming",
    "adobe.json":                  "Adobe",
    "graphicdesign-branding.json": "Design & Branding",
    "image-visualAI.json":         "AI & Visual",
    "motion-vfx-video.json":       "Motion & VFX",
    "ux-product-design.json":      "UX & Product",
}

def clean_text(s):
    s = re.sub(r'http\S+', '', s or '')
    return re.sub(r'\s+', ' ', s).strip()

def norm_bool(v):
    return v.strip().upper() in ("TRUE", "1", "YES")

# ── Load & clean ─────────────────────────────────────────────────────────────

print("Loading CSV…")
with open(INPUT, encoding="utf-8") as f:
    raw = list(csv.DictReader(f))
print(f"  {len(raw)} rows")

posts = []
for r in raw:
    src      = r.get("source_file", "")
    emotion  = r.get("Emotion", "").strip().lower()
    emotion  = EMOTION_NORM.get(emotion, emotion) or "neutral"
    activity = r.get("Creative Activity", "").strip().lower() or "discussion"
    ai_rel   = r.get("AI relation", "").strip().lower()
    ai_rel   = AI_REL_NORM.get(ai_rel, ai_rel) or "none"

    title = (r.get("title") or "").strip()
    body  = clean_text(r.get("body") or "")

    posts.append({
        "id":           r.get("id", ""),
        "title":        title[:200],
        "body_preview": body[:180],
        "subreddit":    r.get("subreddit", ""),
        "category":     SOURCE_CATEGORY.get(src, "Other"),
        "emotion":      emotion,
        "activity":     activity,
        "ai_related":   norm_bool(r.get("AI Related", "")),
        "ai_relation":  ai_rel,
        "adobe_related": norm_bool(r.get("Adobe Related", "")),
        "score":        int(r.get("score") or 0),
        "url":          r.get("url", ""),
        "created_at":   (r.get("createdAt") or "")[:10],
    })

# ── UMAP ─────────────────────────────────────────────────────────────────────

print("\nComputing UMAP layout…")
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import umap as umap_lib

    corpus = [f"{p['title']} {p['body_preview']}" for p in posts]

    print("  TF-IDF vectorizing…")
    vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2),
                          sublinear_tf=True, stop_words="english",
                          min_df=2)
    X = vec.fit_transform(corpus)

    print(f"  Matrix: {X.shape}  — running UMAP (≈1-2 min)…")
    reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.08,
                             n_components=2, metric="cosine",
                             random_state=42, verbose=False,
                             low_memory=False)
    coords = reducer.fit_transform(X)

    for i, p in enumerate(posts):
        p["ux"] = round(float(coords[i, 0]), 4)
        p["uy"] = round(float(coords[i, 1]), 4)

    has_umap = True
    print("  ✓ UMAP done")

except ImportError as e:
    print(f"  ⚠  Skipping UMAP ({e})")
    print("     Install: pip3 install umap-learn scikit-learn")
    for p in posts:
        p["ux"] = 0.0
        p["uy"] = 0.0
    has_umap = False

# ── Subreddit metadata ────────────────────────────────────────────────────────

sub_info = {}
for p in posts:
    s = p["subreddit"]
    if s not in sub_info:
        sub_info[s] = {"name": s, "category": p["category"], "count": 0}
    sub_info[s]["count"] += 1

subreddits = sorted(sub_info.values(), key=lambda x: (x["category"], -x["count"]))

# ── Save ──────────────────────────────────────────────────────────────────────

out = {
    "posts":      posts,
    "subreddits": subreddits,
    "has_umap":   has_umap,
    "total":      len(posts),
}

print(f"\nSaving {OUTPUT}…")
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(out, f, separators=(",", ":"))

size_kb = len(json.dumps(out)) / 1024
print(f"✅  {OUTPUT}  ({size_kb:.0f} KB,  {len(posts)} posts,  UMAP={'yes' if has_umap else 'no'})")
print()
print("─" * 50)
print("Next steps:")
print("  python3 -m http.server 8080")
print("  → open http://localhost:8080")
print("─" * 50)
