import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv


DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_INPUT = "csv/addresses.csv"

TEMPERATURE = 0


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        default="sonnet4.6_poi_results.json",
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
        default=2,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
    )

    return parser.parse_args()


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


def extract_text(response):

    texts = []

    for block in response.content:

        if getattr(block, "type", "") == "text":

            texts.append(block.text)

    return "\n".join(texts).strip()


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
            "Empty JSON response."
        )

    try:

        return json.loads(cleaned)

    except Exception as exc:

        raise ValueError(
            f"Invalid JSON: {exc}\n\n"
            f"Raw response:\n{text}"
        ) from exc


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

    previous_distance = -1

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

        if distance < previous_distance:

            raise ValueError(
                "POIs not sorted by distance."
            )

        previous_distance = distance

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

    return pois


def generate_claude_output(
    client,
    *,
    model,
    prompt,
    max_tokens,
):

    last_error = None

    for attempt in range(3):

        try:

            response = client.messages.create(

                model=model,

                temperature=TEMPERATURE,

                max_tokens=max_tokens,

                system=(
                    "You are a strict JSON generation system. "
                    "Return ONLY valid JSON. "
                    "Do not explain anything. "
                    "Do not use markdown."
                ),

                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            text = extract_text(response)

            usage = {

                "input_tokens":
                response.usage.input_tokens,

                "output_tokens":
                response.usage.output_tokens,
            }

            return {

                "text": text,

                "usage": usage,
            }

        except Exception as exc:

            last_error = exc

            time.sleep(2)

    raise RuntimeError(
        f"Claude request failed: {last_error}"
    )


def save_results(path, payload):

    Path(path).write_text(

        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),

        encoding="utf-8",
    )


def run(args):

    load_dotenv()

    api_key = os.getenv(
        "ANTHROPIC_API_KEY"
    )

    if not api_key:

        raise SystemExit(
            "Missing ANTHROPIC_API_KEY"
        )

    client = Anthropic(
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
            "anthropic",

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

            response = generate_claude_output(

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