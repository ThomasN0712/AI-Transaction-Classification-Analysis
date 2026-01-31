import os
import json
import time
import hashlib
from typing import Dict, Any, List, Tuple

import pandas as pd
from openai import OpenAI

# Load environment variables from .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Read API key from environment (support multiple common names)
from os import getenv
API_KEY = getenv("API_KEY") or getenv("OPENAI_API_KEY") or getenv("API_TOKEN")

INPUT_CSV = "mastersheet.csv"
OUTPUT_CSV = "mastersheet_classified.csv"
CACHE_FILE = "classification_cache.json"

MODEL = "gpt-5-mini"

# How many rows per API call
BATCH_SIZE = 30

# Retry settings
MAX_RETRIES = 5

CATEGORIES = [
    "Income",
    "Transfer",
    "Housing",
    "Transportation",
    "Food",
    "Health",
    "Shopping",
    "Entertainment",
    "Travel",
    "Financial",
    "Education",
    "Gifts",
    "Business",
    "Misc",
]

SYSTEM_PROMPT = f"""
You classify bank transactions using only Description and Amount.

Pick exactly ONE category from this list:
{CATEGORIES}

Return ONLY valid JSON. No markdown. No extra text.

Output format (must match exactly):
{{
  "items": [
    {{
      "row_id": 0,
      "category": "Food",
      "confidence": 0.85
    }}
  ]
}}

Notes:
- confidence must be a number from 0 to 1.
- If unclear, use "Misc" with lower confidence.
- If it looks like paying yourself or moving money between accounts, use "Transfer".
- If positive amount and payroll-like, use "Income".
"""

def hash_key(description: str, amount: str) -> str:
    raw = f"{description}|{amount}".strip().lower()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache: Dict[str, Any]) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def chunk_rows(rows: List[Tuple[int, str, str]], batch_size: int) -> List[List[Tuple[int, str, str]]]:
    return [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

def parse_json_output(text: str) -> Dict[str, Any]:
    t = text.strip()

    # Remove code fences if present
    if t.startswith("```"):
        t = t.replace("```json", "").replace("```", "").strip()

    # If model accidentally added leading/trailing text, try to extract JSON object
    first = t.find("{")
    last = t.rfind("}")
    if first != -1 and last != -1 and last > first:
        t = t[first:last + 1]

    return json.loads(t)

def classify_batch(client: OpenAI, batch: List[Tuple[int, str, str]]) -> List[Dict[str, Any]]:
    payload = {
        "rows": [{"row_id": rid, "description": desc, "amount": amt} for (rid, desc, amt) in batch]
    }

    for attempt in range(MAX_RETRIES):
        try:
            r = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "developer", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
                ]
            )
            data = parse_json_output(r.output_text)
            items = data.get("items", [])
            return items
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)

def read_csv_robust(path: str) -> pd.DataFrame:
    # Try a few common encodings; fallback uses latin1 with replace via open()
    encodings_to_try = ["utf-8-sig", "utf-16", "utf-16-le", "cp1252", "latin1"]
    last_err = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e

    # Final fallback: decode anything and replace bad characters (works on old pandas)
    with open(path, "r", encoding="latin1", errors="replace", newline="") as f:
        return pd.read_csv(f)

def main():
    if not API_KEY:
        raise RuntimeError(
            "API key not found. Set API_KEY (or OPENAI_API_KEY/API_TOKEN) in your environment or .env file."
        )

    client = OpenAI(api_key=API_KEY)

    df = read_csv_robust(INPUT_CSV)

    required = {"Description", "Amount"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must include columns: {sorted(required)}. Found: {df.columns.tolist()}")

    if "Category" not in df.columns:
        df["Category"] = None
    if "Confidence" not in df.columns:
        df["Confidence"] = None

    cache = load_cache()

    # Prepare rows needing classification
    to_classify: List[Tuple[int, str, str]] = []
    for idx, row in df.iterrows():
        desc = "" if pd.isna(row["Description"]) else str(row["Description"])
        amt = "" if pd.isna(row["Amount"]) else str(row["Amount"])
        k = hash_key(desc, amt)

        if k in cache:
            df.at[idx, "Category"] = cache[k]["category"]
            df.at[idx, "Confidence"] = cache[k]["confidence"]
        else:
            to_classify.append((int(idx), desc, amt))

    print(f"Rows total: {len(df)}")
    print(f"Rows cached: {len(df) - len(to_classify)}")
    print(f"Rows to classify: {len(to_classify)}")
    if len(to_classify) == 0:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {OUTPUT_CSV}")
        return

    batches = chunk_rows(to_classify, BATCH_SIZE)

    for bi, batch in enumerate(batches, start=1):
        items = classify_batch(client, batch)

        # Defensive: if model returns fewer items than requested, only apply what we got
        by_row_id = {it["row_id"]: it for it in items if "row_id" in it}

        for (rid, desc, amt) in batch:
            it = by_row_id.get(rid)
            if not it:
                # If missing, mark as Misc with low confidence
                it = {"row_id": rid, "category": "Misc", "confidence": 0.2}

            k = hash_key(desc, amt)
            cache[k] = {"category": it["category"], "confidence": it["confidence"]}

            df.at[rid, "Category"] = it["category"]
            df.at[rid, "Confidence"] = it["confidence"]

        save_cache(cache)
        print(f"Batch {bi}/{len(batches)} complete")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")

if __name__ == "__main__":
    main()


