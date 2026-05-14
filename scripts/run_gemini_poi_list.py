import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


DEFAULT_MODEL = "gemini-3-flash-preview"
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
        default="gemini3flashpreview_poi_results.json",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--api-key",
        default=None,
    )

    parser.add_argument(
        "--prompt",
        default="prompts/POI_list_prompt.txt",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=5,
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


def clean_json_text(text):

    text = str(text or "").strip()

    if not text:
        return ""

    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL,
    )

    if fenced:
        text = fenced.group(1)

    start = text.find("{")

    if start == -1:
        return ""

    text = text[start:]

    return text.strip()


def extract_valid_pois(text):

    pois = []

    # pega somente conteúdo do array
    match = re.search(
        r'"pois"\s*:\s*\[(.*)',
        text,
        flags=re.DOTALL,
    )

    if not match:
        return pois

    content = match.group(1)

    current = ""
    depth = 0
    inside_string = False
    escape = False

    for char in content:

        current += char

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            inside_string = not inside_string
            continue

        if inside_string:
            continue

        if char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:

                candidate = current.strip()

                try:

                    obj = json.loads(candidate)

                    required = {

                        "name",
                        "type",
                        "distance_km",
                        "direction",
                        "confidence",
                    }

                    if (
                        isinstance(obj, dict)
                        and required.issubset(obj.keys())
                    ):

                        pois.append(obj)

                except Exception:
                    pass

                current = ""

    return pois


def repair_json(text):

    text = clean_json_text(text)

    if not text:
        return ""

    # remove null chars
    text = text.replace("\x00", "")

    # remove trailing commas
    text = re.sub(
        r",\s*}",
        "}",
        text,
    )

    text = re.sub(
        r",\s*]",
        "]",
        text,
    )

    # remove fragmentos tipo:
    # {]}
    text = re.sub(
        r"\{\s*]",
        "{}",
        text,
    )

    # remove objetos truncados finais
    text = re.sub(
        r',?\s*\{\s*"[^"]*$',
        "",
        text,
        flags=re.DOTALL,
    )

    # balanceamento
    open_brackets = text.count("[")
    close_brackets = text.count("]")

    if close_brackets < open_brackets:
        text += "]" * (
            open_brackets - close_brackets
        )

    open_braces = text.count("{")
    close_braces = text.count("}")

    if close_braces < open_braces:
        text += "}" * (
            open_braces - close_braces
        )

    return text


def parse_model_json(text):

    cleaned = clean_json_text(text)

    if not cleaned:

        raise ValueError(
            "Model returned empty JSON."
        )

    # tentativa original
    try:

        return json.loads(cleaned)

    except Exception:

        pass

    # tentativa reparada
    repaired = repair_json(cleaned)

    try:

        return json.loads(repaired)

    except Exception:

        pass

    # recuperação robusta
    pois = extract_valid_pois(repaired)

    if pois:

        return {
            "pois": pois
        }

    raise ValueError(
        "Invalid JSON after repair. "
        f"Preview: {repaired[:1200]}"
    )


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

    # aceita parcial
    if len(pois) < 1:

        raise ValueError(
            "No valid POIs found."
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

    validated_pois = []

    for poi in pois:

        try:

            if not isinstance(
                poi,
                dict,
            ):
                continue

            name = poi.get("name")

            if (
                not isinstance(name, str)
                or not name.strip()
            ):
                continue

            poi_type = poi.get("type")

            if poi_type not in allowed_types:
                continue

            distance = poi.get(
                "distance_km"
            )

            if not isinstance(
                distance,
                (int, float),
            ):
                continue

            if distance < 0:
                continue

            if distance > 2.0:
                continue

            direction = poi.get(
                "direction"
            )

            if direction not in allowed_directions:
                continue

            confidence = poi.get(
                "confidence"
            )

            if confidence not in allowed_confidence:
                continue

            if distance < previous_distance:
                continue

            previous_distance = distance

            validated_pois.append(poi)

        except Exception:
            continue

    if not validated_pois:

        raise ValueError(
            "No valid POIs after validation."
        )

    return validated_pois


def generate_gemini_output(
    prompt,
    *,
    api_key,
    model,
    max_tokens,
):

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:generateContent"
    )

    payload = {

        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": prompt
                    }
                ],
            }
        ],

        "generationConfig": {

            "temperature":
            TEMPERATURE,

            "maxOutputTokens":
            max_tokens,

            "responseMimeType":
            "application/json",

            "responseSchema": {

                "type": "OBJECT",

                "properties": {

                    "pois": {

                        "type": "ARRAY",

                        "items": {

                            "type": "OBJECT",

                            "properties": {

                                "name": {
                                    "type": "STRING"
                                },

                                "type": {
                                    "type": "STRING"
                                },

                                "distance_km": {
                                    "type": "NUMBER"
                                },

                                "direction": {
                                    "type": "STRING"
                                },

                                "confidence": {
                                    "type": "STRING"
                                },
                            },

                            "required": [

                                "name",
                                "type",
                                "distance_km",
                                "direction",
                                "confidence",
                            ],
                        },
                    }
                },

                "required": [
                    "pois"
                ],
            },
        },
    }

    last_error = None

    for attempt in range(8):

        try:

            response = requests.post(

                url,

                headers={
                    "Content-Type":
                    "application/json",

                    "x-goog-api-key":
                    api_key,
                },

                json=payload,

                timeout=180,
            )

            response.raise_for_status()

            data = response.json()

            candidates = data.get(
                "candidates",
                []
            )

            if not candidates:

                raise RuntimeError(
                    "Gemini returned no candidates."
                )

            parts = (

                candidates[0]

                .get("content", {})

                .get("parts", [])
            )

            text = "".join(

                part.get("text", "")

                for part in parts
            ).strip()

            if not text:

                raise RuntimeError(
                    "Empty Gemini response."
                )

            usage = data.get(
                "usageMetadata",
                {}
            )

            return {

                "text": text,

                "usage": {

                    "input_tokens":
                    usage.get(
                        "promptTokenCount"
                    ),

                    "output_tokens":
                    usage.get(
                        "candidatesTokenCount"
                    ),
                },
            }

        except Exception as exc:

            last_error = exc

            print(
                f"Retry {attempt + 1}/8 failed: "
                f"{exc}"
            )

            time.sleep(
                min(2 ** attempt, 30)
            )

    raise RuntimeError(
        f"Gemini request failed: "
        f"{last_error}"
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

    api_key = (

        args.api_key

        or os.getenv(
            "GEMINI_API_KEY"
        )
    )

    if not api_key:

        raise SystemExit(
            "Missing GEMINI_API_KEY."
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
            "gemini",

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

            response = generate_gemini_output(

                prompt,

                api_key=api_key,

                model=args.model,

                max_tokens=args.max_tokens,
            )

            prediction = None
            validation_error = None

            for _ in range(3):

                try:

                    prediction = parse_model_json(
                        response["text"]
                    )

                    validated = validate_output(
                        prediction
                    )

                    break

                except Exception as exc:

                    validation_error = exc

                    time.sleep(1)

            else:

                raise validation_error

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