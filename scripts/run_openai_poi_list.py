import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# CONFIG
# =========================================================

DEFAULT_MODEL = "gpt-4.1"
DEFAULT_INPUT = "csv/addresses.csv"

TEMPERATURE = 0


# =========================================================
# ARGUMENTS
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        default="openai_poi_results.json",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--prompt",
        default="prompts/POI_list_prompt.txt",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
    )

    return parser.parse_args()


# =========================================================
# PROMPT
# =========================================================

def load_prompt(path):

    return Path(path).read_text(
        encoding="utf-8"
    )


def build_prompt(
    template,
    row,
):

    values = {

        "{address}":
        row.get("address", ""),

        "{city}":
        row.get("city", ""),

        "{latitude}":
        row.get("latitude", ""),

        "{longitude}":
        row.get("longitude", ""),
    }

    prompt = template

    for placeholder, value in values.items():

        prompt = prompt.replace(
            placeholder,
            str(value),
        )

    return prompt


def build_system_prompt():

    return """
You must return ONLY a valid JSON object.

Do not explain anything.
Do not include reasoning.
Do not use markdown.

Schema:

{
  "pois": [
    {
      "name": "string",
      "type": "restaurant|school|hospital|park|retail|transport|landmark|residential|industrial|religious",
      "distance_km": 0.1,
      "direction": "N|NE|NW|S|SE|SW|E|W|unknown",
      "confidence": "low|medium|high"
    }
  ]
}

Rules:
- Exactly 10 POIs
- distance_km <= 2.0
- Sorted by distance ascending
- Output must start with {
- Output must end with }
"""


# =========================================================
# JSON PARSING
# =========================================================

def clean_json_text(text):

    text = str(text or "").strip()

    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL,
    )

    if fenced:

        text = fenced.group(1)

    return text.strip()


def parse_model_json(text):

    cleaned = clean_json_text(text)

    if not cleaned:

        raise ValueError(
            "Empty response."
        )

    match = re.search(
        r"\{.*\}",
        cleaned,
        flags=re.DOTALL,
    )

    if not match:

        raise ValueError(
            f"No JSON object found.\n\n"
            f"Raw response:\n{text}"
        )

    json_text = match.group(0)

    try:

        return json.loads(json_text)

    except Exception as exc:

        raise ValueError(
            f"Invalid JSON: {exc}\n\n"
            f"Extracted JSON:\n{json_text}"
        ) from exc


# =========================================================
# VALIDATION
# =========================================================

def validate_output(prediction):

    if not isinstance(prediction, dict):

        raise ValueError(
            "Prediction must be object."
        )

    pois = prediction.get("pois")

    if not isinstance(pois, list):

        raise ValueError(
            "pois must be list."
        )

    if len(pois) != 10:

        raise ValueError(
            f"Expected 10 POIs, "
            f"got {len(pois)}"
        )

    allowed_types = {

        "restaurant",
        "school",
        "hospital",
        "park",
        "retail",
        "transport",
        "landmark",
        "residential",
        "industrial",
        "religious",
    }

    allowed_directions = {

        "N",
        "NE",
        "NW",
        "S",
        "SE",
        "SW",
        "E",
        "W",
        "unknown",
    }

    allowed_confidence = {

        "low",
        "medium",
        "high",
    }

    validated = []

    for poi in pois:

        if not isinstance(
            poi,
            dict,
        ):

            raise ValueError(
                "POI must be object."
            )

        name = poi.get("name")

        if not isinstance(name, str) or not name.strip():

            raise ValueError(
                "Invalid POI name."
            )

        poi_type = poi.get("type")

        if poi_type not in allowed_types:

            raise ValueError(
                f"Invalid type: {poi_type}"
            )

        distance = poi.get(
            "distance_km"
        )

        if not isinstance(
            distance,
            (int, float),
        ):

            raise ValueError(
                "distance_km must be numeric."
            )

        if distance < 0:

            raise ValueError(
                "Negative distance."
            )

        if distance > 2.0:

            raise ValueError(
                "distance_km > 2.0"
            )

        direction = poi.get(
            "direction"
        )

        if direction not in allowed_directions:

            raise ValueError(
                f"Invalid direction: {direction}"
            )

        confidence = poi.get(
            "confidence"
        )

        if confidence not in allowed_confidence:

            raise ValueError(
                f"Invalid confidence: {confidence}"
            )

        validated.append({

            "name":
            name.strip(),

            "type":
            poi_type,

            "distance_km":
            float(distance),

            "direction":
            direction,

            "confidence":
            confidence,
        })

    validated.sort(
        key=lambda x: x["distance_km"]
    )

    return validated


# =========================================================
# OPENAI API
# =========================================================

def generate_openai_output(
    client,
    *,
    model,
    prompt,
    max_tokens,
):

    last_error = None

    system_prompt = build_system_prompt()

    for attempt in range(3):

        try:

            response = client.chat.completions.create(

                model=model,

                temperature=TEMPERATURE,

                max_completion_tokens=max_tokens,

                response_format={
                    "type": "json_object"
                },

                messages=[

                    {
                        "role": "system",
                        "content": system_prompt,
                    },

                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            text = response.choices[0].message.content

            text = str(text or "").strip()

            print(
                "\n================ RAW RESPONSE ================\n"
            )

            print(text[:3000])

            print(
                "\n==============================================\n"
            )

            if not text:

                raise ValueError(
                    "Model returned empty response."
                )

            usage = {

                "input_tokens":
                response.usage.prompt_tokens,

                "output_tokens":
                response.usage.completion_tokens,
            }

            return {

                "text": text,

                "usage": usage,
            }

        except Exception as exc:

            last_error = exc

            print(
                f"\nAttempt {attempt + 1} failed:"
            )

            print(exc)

            time.sleep(2)

    raise RuntimeError(
        f"OpenAI request failed: {last_error}"
    )


# =========================================================
# SAVE
# =========================================================

def save_results(path, payload):

    Path(path).write_text(

        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),

        encoding="utf-8",
    )


# =========================================================
# MAIN
# =========================================================

def run(args):

    load_dotenv()

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:

        raise SystemExit(
            "Missing OPENAI_API_KEY"
        )

    client = OpenAI(
        api_key=api_key
    )

    prompt_template = load_prompt(
        args.prompt
    )

    df = pd.read_csv(args.input)

    if args.limit is not None:

        df = df.head(args.limit)

    results = {

        "_metadata": {

            "provider":
            "openai",

            "model":
            args.model,

            "temperature":
            TEMPERATURE,
        },

        "results": [],
    }

    for index, row in df.iterrows():

        address = row.get(
            "address",
            "",
        )

        print(
            f"[{index + 1}/{len(df)}] "
            f"{address}"
        )

        try:

            prompt = build_prompt(
                prompt_template,
                row,
            )

            response = generate_openai_output(

                client=client,

                model=args.model,

                prompt=prompt,

                max_tokens=args.max_tokens,
            )

            prediction = parse_model_json(
                response["text"]
            )

            validated = validate_output(
                prediction
            )

            result = {

                "city":
                row.get("city"),

                "address":
                address,

                "latitude":
                row.get("latitude"),

                "longitude":
                row.get("longitude"),

                "pois":
                validated,

                "usage":
                response["usage"],
            }

        except Exception as exc:

            result = {

                "city":
                row.get("city"),

                "address":
                address,

                "latitude":
                row.get("latitude"),

                "longitude":
                row.get("longitude"),

                "error":
                str(exc),
            }

            print("\nERROR:")
            print(exc)

        results["results"].append(
            result
        )

        save_results(
            args.output,
            results,
        )

        if args.delay > 0:

            time.sleep(args.delay)

    save_results(
        args.output,
        results,
    )

    print(
        f"\nFinished. "
        f"Saved at: {args.output}"
    )


def main():

    run(parse_args())


if __name__ == "__main__":

    main()