import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_INPUT = "csv/addresses.csv"

TEMPERATURE = 0

BENCHMARK_VERSION = "2.0"
PROMPT_VERSION = "v2"
TAXONOMY_VERSION = "v3"


COUNTRY_ALIASES = {
    "brasil": "Brazil",
    "brazil": "Brazil",
    "canada": "Canada",
    "canadá": "Canada",
    "eua": "United States",
    "estados unidos": "United States",
    "united states": "United States",
    "usa": "United States",
    "reino unido": "United Kingdom",
    "united kingdom": "United Kingdom",
    "uk": "United Kingdom",
    "países baixos": "Netherlands",
    "paises baixos": "Netherlands",
    "netherlands": "Netherlands",
}


LEXICAL_FEATURE_NORMALIZATION = {
    "parks": "park",
    "rivers": "river",
    "lakes": "lake",
    "forests": "forest",
    "beaches": "beach",
}


def parse_args():

    parser = argparse.ArgumentParser(
        description="Run DeepSeek geographic feature inference."
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        default=None,
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
        default="prompts/address_user_prompt.txt",
    )

    parser.add_argument(
        "--country-column",
        default=None,
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=3,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3500,
    )

    parser.add_argument(
        "--retry-errors-from",
        default=None,
    )

    return parser.parse_args()


def load_prompt(path):

    return Path(path).read_text(
        encoding="utf-8"
    )


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


def default_output_path(model):

    safe_model = (
        normalize_label(model)
        .replace("_", "-")
    )

    return (
        f"geographic_features_"
        f"{safe_model}.json"
    )


def clean_json_text(text):

    text = str(text or "").strip()

    if not text:
        return ""

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL,
    )

    if fenced:

        text = fenced.group(1).strip()

    start = text.find("{")

    if start == -1:
        return ""

    text = text[start:]

    end = text.rfind("}")

    if end != -1:
        text = text[: end + 1]

    return text.strip()


def try_fix_truncated_json(text):

    text = text.strip()

    open_braces = text.count("{")
    close_braces = text.count("}")

    open_brackets = text.count("[")
    close_brackets = text.count("]")

    if close_brackets < open_brackets:

        text += "]" * (
            open_brackets - close_brackets
        )

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

    try:

        return json.loads(cleaned)

    except Exception:

        repaired = try_fix_truncated_json(
            cleaned
        )

        try:

            return json.loads(repaired)

        except Exception as exc:

            preview = repaired[:500]

            raise ValueError(
                f"Invalid JSON returned by model: "
                f"{exc}. "
                f"Preview: {preview}"
            ) from exc


def normalize_geographic_features(prediction):

    if not isinstance(prediction, dict):
        return prediction

    features = prediction.get(
        "geographic_features"
    )

    if not isinstance(features, list):
        return prediction

    for item in features:

        if (
            not isinstance(item, dict)
            or "feature" not in item
        ):
            continue

        normalized = normalize_label(
            item["feature"]
        )

        item["feature"] = (
            LEXICAL_FEATURE_NORMALIZATION.get(
                normalized,
                normalized,
            )
        )

    return prediction


def validate_prediction(prediction):

    if not isinstance(prediction, dict):

        raise ValueError(
            "Prediction is not a JSON object."
        )

    features = prediction.get(
        "geographic_features"
    )

    if not isinstance(features, list):

        raise ValueError(
            "Prediction must contain "
            "geographic_features as a list."
        )

    if len(features) > 5:

        raise ValueError(
            "Too many geographic features."
        )

    for feature in features:

        if not isinstance(feature, dict):

            raise ValueError(
                "Each geographic feature "
                "must be an object."
            )

        if not feature.get("feature"):

            raise ValueError(
                "Each geographic feature "
                "must contain a feature name."
            )

        osm_tags = feature.get("osm_tags")

        if not isinstance(
            osm_tags,
            (dict, list),
        ):

            raise ValueError(
                "Each geographic feature "
                "must contain osm_tags "
                "as object or list."
            )

        if isinstance(osm_tags, list):

            for item in osm_tags:

                if not isinstance(item, dict):

                    raise ValueError(
                        "Each item inside "
                        "osm_tags must be object."
                    )

    return prediction


def get_row_value(row, *names):

    for name in names:

        if (
            name in row
            and pd.notna(row[name])
        ):
            return row[name]

    return ""


def infer_country(
    row,
    country_column=None,
):

    if country_column:

        country = get_row_value(
            row,
            country_column,
        )

        if country:

            return str(country).strip()

    address = str(
        get_row_value(
            row,
            "address",
            "Address",
        )
    ).lower()

    parts = [

        part.strip(" .")

        for part in address.split(",")
    ]

    for part in reversed(parts):

        if part in COUNTRY_ALIASES:

            return COUNTRY_ALIASES[part]

    for alias, country in COUNTRY_ALIASES.items():

        if re.search(
            rf"\b{re.escape(alias)}\b",
            address,
        ):

            return country

    return "unknown"


def build_prompt(
    prompt_template,
    row,
):

    values = {

        "{address}":
        get_row_value(
            row,
            "address",
            "Address",
        ),

        "{city}":
        get_row_value(
            row,
            "city",
            "City",
        ),

        "{latitude}":
        get_row_value(
            row,
            "latitude",
            "lat",
            "Latitude",
            "Lat",
        ),

        "{longitude}":
        get_row_value(
            row,
            "longitude",
            "lon",
            "lng",
            "Longitude",
            "Lon",
            "Lng",
        ),
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
        "- Return ONLY one complete JSON object\n"
        "- Do not use markdown\n"
        "- Do not explain\n"
        "- Do not output comments\n"
        "- Do not output reasoning\n"
        "- Do not use <think>\n"
        "- Do not return partial JSON\n"
        "- Omit uncertain features\n"
        "- Return at most 5 geographic features"
    )


def generate_deepseek_text(
    client,
    prompt,
    *,
    model,
    max_tokens,
):

    last_error = None

    for attempt in range(3):

        try:

            response = client.chat.completions.create(

                model=model,

                temperature=TEMPERATURE,

                max_tokens=max_tokens,

                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict JSON API. "
                            "Return ONLY valid JSON. "
                            "Do not explain. "
                            "Do not use markdown. "
                            "Do not use <think>. "
                            "Do not include reasoning."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            message = response.choices[0].message

            text = getattr(
                message,
                "content",
                None,
            )

            if isinstance(text, list):

                text = "\n".join(

                    item.get("text", "")

                    if isinstance(item, dict)

                    else str(item)

                    for item in text
                )

            text = str(text or "").strip()

            if not text.strip():

                raise RuntimeError(
                    "Empty model response."
                )

            usage = {

                "input_tokens":
                getattr(
                    response.usage,
                    "prompt_tokens",
                    None,
                ),

                "output_tokens":
                getattr(
                    response.usage,
                    "completion_tokens",
                    None,
                ),
            }

            return {

                "text":
                text,

                "usage":
                usage,
            }

        except Exception as exc:

            last_error = exc

            print(
                f"Retry {attempt + 1}/3 failed: "
                f"{exc}"
            )

            time.sleep(2)

    raise RuntimeError(
        f"DeepSeek request failed: "
        f"{last_error}"
    )


def result_key(result):

    return (

        str(result.get("city", "")),

        str(result.get("address", "")),

        str(result.get("latitude", "")),

        str(result.get("longitude", "")),
    )


def row_key(row):

    return (

        str(
            get_row_value(
                row,
                "city",
                "City",
            )
        ),

        str(
            get_row_value(
                row,
                "address",
                "Address",
            )
        ),

        str(
            get_row_value(
                row,
                "latitude",
                "lat",
                "Latitude",
                "Lat",
            )
        ),

        str(
            get_row_value(
                row,
                "longitude",
                "lon",
                "lng",
                "Longitude",
                "Lon",
                "Lng",
            )
        ),
    )


def load_previous_results(path):

    previous_path = Path(path)

    if not previous_path.exists():

        raise SystemExit(
            f"Previous output not found: "
            f"{previous_path}"
        )

    data = json.loads(
        previous_path.read_text(
            encoding="utf-8"
        )
    )

    if "results_by_country" not in data:

        raise SystemExit(
            "Previous output must contain "
            "results_by_country."
        )

    return data


def error_keys_from_previous(
    previous_results,
):

    keys = set()

    for results in previous_results.get(
        "results_by_country",
        {},
    ).values():

        for result in results:

            if (
                isinstance(result, dict)
                and "error" in result
            ):

                keys.add(
                    result_key(result)
                )

    return keys


def replace_result(
    grouped_results,
    country,
    new_result,
):

    results = grouped_results[
        "results_by_country"
    ].setdefault(
        country,
        [],
    )

    new_key = result_key(new_result)

    for index, existing in enumerate(results):

        if result_key(existing) == new_key:

            results[index] = new_result

            return

    results.append(new_result)


def save_results(path, payload):

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(

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
            "DEEPSEEK_API_KEY"
        )
    )

    if not api_key:

        raise SystemExit(
            "Missing DEEPSEEK_API_KEY."
        )

    client = OpenAI(

        api_key=api_key,

        base_url=(
            "https://api.deepseek.com"
        ),
    )

    output_path = (
        args.output
        or default_output_path(
            args.model
        )
    )

    prompt_template = load_prompt(
        args.prompt
    )

    df = pd.read_csv(args.input)

    retry_keys = None
    previous_results = None

    if args.retry_errors_from:

        previous_results = load_previous_results(
            args.retry_errors_from
        )

        retry_keys = error_keys_from_previous(
            previous_results
        )

        df = df[
            df.apply(
                lambda row:
                row_key(row) in retry_keys,
                axis=1,
            )
        ]

        print(
            f"Retry mode: "
            f"{len(df)} rows."
        )

    if args.limit is not None:

        df = df.head(args.limit)

    grouped_results = previous_results or {

        "_metadata": {

            "provider":
            "deepseek",

            "model":
            args.model,

            "temperature":
            TEMPERATURE,

            "benchmark_version":
            BENCHMARK_VERSION,

            "prompt_version":
            PROMPT_VERSION,

            "taxonomy_version":
            TAXONOMY_VERSION,

            "uses_external_tools":
            False,
        },

        "results_by_country": {},
    }

    save_results(
        output_path,
        grouped_results,
    )

    for index, row in df.iterrows():

        address = get_row_value(
            row,
            "address",
            "Address",
        )

        country = infer_country(
            row,
            args.country_column,
        )

        print(
            f"[{index + 1}/{len(df)}] "
            f"{country} - {address}"
        )

        try:

            prompt = build_prompt(
                prompt_template,
                row,
            )

            response = generate_deepseek_text(

                client=client,

                prompt=prompt,

                model=args.model,

                max_tokens=args.max_tokens,
            )

            prediction = (
                validate_prediction(

                    normalize_geographic_features(

                        parse_model_json(
                            response["text"]
                        )
                    )
                )
            )

            result = {

                "city":
                get_row_value(
                    row,
                    "city",
                    "City",
                ),

                "address":
                address,

                "latitude":
                get_row_value(
                    row,
                    "latitude",
                    "lat",
                    "Latitude",
                    "Lat",
                ),

                "longitude":
                get_row_value(
                    row,
                    "longitude",
                    "lon",
                    "lng",
                    "Longitude",
                    "Lon",
                    "Lng",
                ),

                "prediction":
                prediction,

                "usage":
                response["usage"],
            }

        except Exception as exc:

            result = {

                "city":
                get_row_value(
                    row,
                    "city",
                    "City",
                ),

                "address":
                address,

                "latitude":
                get_row_value(
                    row,
                    "latitude",
                    "lat",
                    "Latitude",
                    "Lat",
                ),

                "longitude":
                get_row_value(
                    row,
                    "longitude",
                    "lon",
                    "lng",
                    "Longitude",
                    "Lon",
                    "Lng",
                ),

                "error":
                str(exc),
            }

        if retry_keys is None:

            grouped_results[
                "results_by_country"
            ].setdefault(
                country,
                [],
            ).append(result)

        else:

            replace_result(
                grouped_results,
                country,
                result,
            )

        save_results(
            output_path,
            grouped_results,
        )

        if args.delay > 0:

            time.sleep(args.delay)

    save_results(
        output_path,
        grouped_results,
    )

    print(
        f"\nFinished. "
        f"Saved at: {output_path}"
    )


def main():

    run(parse_args())


if __name__ == "__main__":

    main()