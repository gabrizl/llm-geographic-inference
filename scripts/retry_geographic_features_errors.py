import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import requests

from src.env_loader import load_dotenv
from src.prompt_loader import load_prompt


DEFAULT_MODEL = "gemini-3.1-flash-lite"

TEMPERATURE = 0

MAX_RETRIES_PER_ITEM = 5


LEXICAL_FEATURE_NORMALIZATION = {
    "parks": "park",
    "rivers": "river",
    "lakes": "lake",
    "forests": "forest",
    "beaches": "beach",
}


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Retry invalid POI generations "
            "and replace them in original JSON."
        )
    )

    parser.add_argument(
        "--json-file",
        required=True,
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
        default="address_user_prompt.txt",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=4,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    return parser.parse_args()


def normalize_label(value):

    value = str(value or "").strip().lower()

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        char
        for char in value
        if not unicodedata.combining(char)
    )

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    ).strip("_")

    return value


def clean_json_text(text):

    text = (text or "").strip()

    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL,
    )

    if fenced:
        text = fenced.group(1).strip()

    start = text.find("{")

    if start == -1:
        return text

    text = text[start:]

    end = text.rfind("}")

    if end != -1:
        text = text[: end + 1]

    return text.strip()


def try_fix_truncated_json(text):

    if not text:
        return text

    text = text.strip()

    text = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace("```", "")

    first = text.find("{")

    if first != -1:
        text = text[first:]

    last = text.rfind("}")

    if last != -1:
        text = text[: last + 1]

    text = re.sub(
        r'(\]\s*\})\s*(\{)',
        r'\1,\2',
        text,
        flags=re.MULTILINE,
    )

    text = re.sub(
        r'(\})\s*(\{)',
        r'\1,\2',
        text,
        flags=re.MULTILINE,
    )

    while "}}}" in text:
        text = text.replace(
            "}}}",
            "}}",
        )

    while "]]]" in text:
        text = text.replace(
            "]]]",
            "]]",
        )

    open_curly = text.count("{")
    close_curly = text.count("}")

    open_square = text.count("[")
    close_square = text.count("]")

    if close_square < open_square:

        text += "]" * (
            open_square - close_square
        )

    if close_curly < open_curly:

        text += "}" * (
            open_curly - close_curly
        )

    return text


def parse_model_json(text):

    cleaned = clean_json_text(text)

    if not cleaned:

        raise ValueError(
            "Empty JSON response."
        )

    try:

        return json.loads(cleaned)

    except json.JSONDecodeError:
        pass

    repaired = try_fix_truncated_json(
        cleaned
    )

    try:

        return json.loads(repaired)

    except json.JSONDecodeError as exc:

        raise ValueError(
            f"Invalid JSON returned by model: "
            f"{exc}. "
            f"Preview: {repaired[:500]}"
        )


def retry_delay_seconds(
    response,
    *,
    fallback,
):

    retry_after = response.headers.get(
        "Retry-After"
    )

    if retry_after:

        try:

            return max(
                float(retry_after),
                fallback,
            )

        except ValueError:
            pass

    return fallback


def raise_for_gemini_status(response):

    try:

        response.raise_for_status()

    except requests.HTTPError as exc:

        message = (
            f"Gemini API returned HTTP "
            f"{response.status_code}"
        )

        try:

            error = response.json().get(
                "error",
                {},
            )

            if error.get("message"):

                message = (
                    f"{message}: "
                    f"{error['message']}"
                )

        except ValueError:

            if response.text:

                message = (
                    f"{message}: "
                    f"{response.text[:300]}"
                )

        raise RuntimeError(message) from exc


def generate_gemini_text(
    prompt,
    *,
    api_key,
    model,
    max_tokens,
    timeout=120,
):

    url = (
        f"https://generativelanguage.googleapis.com/"
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
        },
    }

    response = None

    for attempt in range(6):

        try:

            response = requests.post(

                url,

                headers={
                    "content-type":
                    "application/json",

                    "x-goog-api-key":
                    api_key,
                },

                json=payload,

                timeout=timeout,
            )

        except requests.RequestException as exc:

            if attempt == 5:

                raise RuntimeError(
                    f"Gemini request failed: {exc}"
                )

            delay = min(
                2 ** attempt,
                30,
            )

            print(
                f"Request failed. "
                f"Retrying in {delay}s..."
            )

            time.sleep(delay)

            continue

        if response.status_code not in {
            429,
            500,
            502,
            503,
            504,
        }:
            break

        if attempt < 5:

            delay = retry_delay_seconds(
                response,
                fallback=2 ** attempt,
            )

            print(
                f"Gemini returned "
                f"HTTP {response.status_code}. "
                f"Waiting {delay:.1f}s..."
            )

            time.sleep(delay)

    if response is None:

        raise RuntimeError(
            "Gemini request was not sent."
        )

    raise_for_gemini_status(response)

    data = response.json()

    candidates = data.get(
        "candidates"
    ) or []

    if not candidates:

        raise RuntimeError(
            "Gemini returned no candidates."
        )

    content = candidates[0].get(
        "content"
    ) or {}

    parts = content.get(
        "parts"
    ) or []

    text = "".join(

        part.get("text", "")

        for part in parts

        if isinstance(part, dict)
    )

    if not text.strip():

        raise RuntimeError(
            "Empty Gemini response."
        )

    return text


def normalize_pois(pois):

    if not isinstance(pois, list):
        return pois

    for poi in pois:

        if not isinstance(poi, dict):
            continue

        if "type" in poi:

            normalized = normalize_label(
                poi["type"]
            )

            poi["type"] = (
                LEXICAL_FEATURE_NORMALIZATION.get(
                    normalized,
                    normalized,
                )
            )

    return pois


def validate_prediction(prediction):

    if not isinstance(
        prediction,
        dict,
    ):

        raise ValueError(
            "Prediction is not a JSON object."
        )

    pois = prediction.get("pois")

    if not isinstance(pois, list):

        raise ValueError(
            "Prediction must contain "
            "'pois' as a list."
        )

    return prediction


def parse_distance(value):

    if value is None:
        return None

    value = str(value)

    value = re.sub(
        r"[^0-9.]",
        "",
        value,
    )

    if not value:
        return None

    return float(value)


def is_sorted_by_distance(pois):

    distances = []

    for poi in pois:

        if not isinstance(poi, dict):
            return False

        distance = parse_distance(
            poi.get("distance_km")
        )

        if distance is None:
            continue

        distances.append(distance)

    return distances == sorted(distances)


def should_retry_prediction(
    prediction
):

    if not isinstance(
        prediction,
        dict,
    ):
        return True

    pois = prediction.get("pois")

    if not isinstance(pois, list):
        return True

    if len(pois) == 0:
        return True

    for poi in pois:

        if not isinstance(poi, dict):
            return True

        if not poi.get("name"):
            return True

        if not poi.get("distance_km"):
            return True

    if not is_sorted_by_distance(
        pois
    ):
        return True

    return False


def build_prompt(
    prompt_template,
    item,
):

    values = {

        "{address}":
        item.get("address", ""),

        "{city}":
        item.get("city", ""),

        "{latitude}":
        item.get("latitude", ""),

        "{longitude}":
        item.get("longitude", ""),
    }

    prompt = prompt_template

    for placeholder, value in values.items():

        prompt = prompt.replace(
            placeholder,
            str(value),
        )

    return (
        f"{prompt}\n\n"

        "STRICT RULES:\n"

        "- Return exactly ONE valid JSON object\n"
        "- Do not use markdown\n"
        "- Do not explain\n"
        "- Do not output comments\n"
        "- Do not output reasoning\n"
        "- Do not return partial JSON\n"
        "- Return at most 10 POIs\n"
        "- Sort POIs by distance_km ascending\n"
        "- distance_km must be numeric\n"
        "- Do not hallucinate fake locations\n"
    )


def has_error(item):

    return (
        "error" in item
    )


def save_json(path, data):

    Path(path).write_text(

        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),

        encoding="utf-8",
    )


def process_item(
    item,
    *,
    api_key,
    model,
    max_tokens,
    prompt_template,
):

    last_error = None

    for attempt in range(
        MAX_RETRIES_PER_ITEM
    ):

        try:

            raw_text = generate_gemini_text(

                build_prompt(
                    prompt_template,
                    item,
                ),

                api_key=api_key,

                model=model,

                max_tokens=max_tokens,
            )

            prediction = parse_model_json(
                raw_text
            )

            prediction = validate_prediction(
                prediction
            )

            prediction["pois"] = (
                normalize_pois(
                    prediction["pois"]
                )
            )

            if should_retry_prediction(
                prediction
            ):

                raise ValueError(
                    "POIs not sorted by distance."
                )

            item.pop("error", None)

            item["pois"] = prediction[
                "pois"
            ]

            return item

        except Exception as exc:

            last_error = exc

            print(
                f"Retry "
                f"{attempt + 1}/"
                f"{MAX_RETRIES_PER_ITEM} "
                f"failed: {exc}"
            )

            time.sleep(
                min(
                    2 ** attempt,
                    30,
                )
            )

    raise RuntimeError(
        f"Failed after retries: "
        f"{last_error}"
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
            "Missing Gemini API key."
        )

    prompt_template = load_prompt(
        args.prompt
    )

    json_path = Path(
        args.json_file
    )

    data = json.loads(

        json_path.read_text(
            encoding="utf-8"
        )
    )

    iterable = []

    if isinstance(data, list):

        iterable.extend(data)

    elif isinstance(data, dict):

        if isinstance(
            data.get("results"),
            list,
        ):

            iterable.extend(
                data["results"]
            )

        results_by_country = data.get(
            "results_by_country"
        )

        if isinstance(
            results_by_country,
            dict,
        ):

            for results in (
                results_by_country.values()
            ):

                if isinstance(
                    results,
                    list,
                ):

                    iterable.extend(
                        results
                    )

    failed_items = []

    error_items = []

    for item in iterable:

        if not isinstance(item, dict):
            continue

        has_prediction_problem = False

        if "pois" in item:

            try:

                prediction = {
                    "pois": item["pois"]
                }

                validate_prediction(
                    prediction
                )

                if should_retry_prediction(
                    prediction
                ):
                    has_prediction_problem = True

            except Exception:

                has_prediction_problem = True

        if (
            has_error(item)
            or has_prediction_problem
        ):

            error_items.append(item)

    if args.limit is not None:

        error_items = error_items[
            : args.limit
        ]

    print(
        f"TOTAL ITERABLE: "
        f"{len(iterable)}"
    )

    print(
        f"Found "
        f"{len(error_items)} "
        f"items to retry."
    )

    for index, item in enumerate(
        error_items
    ):

        print(
            f"[{index + 1}/"
            f"{len(error_items)}] "
            f"{item.get('address')}"
        )

        try:

            process_item(

                item,

                api_key=api_key,

                model=args.model,

                max_tokens=args.max_tokens,

                prompt_template=prompt_template,
            )

            print("SUCCESS")

        except Exception as exc:

            item["error"] = str(exc)

            failed_items.append(item)

            print(
                f"ERROR: {exc}"
            )

        save_json(
            json_path,
            data,
        )

        if args.delay > 0:

            time.sleep(
                args.delay
            )

    if failed_items:

        failed_path = (
            json_path.parent
            / "failed_retry_items.json"
        )

        save_json(
            failed_path,
            failed_items,
        )

        print(
            f"\nSaved failed items to: "
            f"{failed_path}"
        )

    print(
        f"\nFinished.\n"
        f"Updated original file: "
        f"{json_path}"
    )


def main():

    run(parse_args())


if __name__ == "__main__":

    main()