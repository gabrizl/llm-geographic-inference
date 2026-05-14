import argparse
import json
import time
from pathlib import Path

import requests


DEFAULT_RADIUS_METERS = 200

MIN_MATCHES_FOR_REAL_FEATURE = 3

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]


TAG_KEYS = {
    "building",
    "highway",
    "landuse",
    "leisure",
    "natural",
    "office",
    "public_transport",
    "railway",
    "waterway",
    "tourism",
}


FEATURE_EQUIVALENCE = {

    "commercial_area": [
        {"landuse": "commercial"},
        {"building": "commercial"},
        {"building": "retail"},
    ],

    "residential_area": [
        {"landuse": "residential"},
        {"building": "residential"},
        {"building": "apartments"},
        {"building": "house"},
    ],

    "mixed_use_area": [
        {"building": "mixed_use"},
    ],

    "industrial_area": [
        {"landuse": "industrial"},
        {"building": "industrial"},
        {"man_made": "works"},
    ],

    "transportation_corridor": [
        {"highway": "motorway"},
        {"highway": "trunk"},
        {"highway": "primary"},
        {"highway": "secondary"},
        {"highway": "tertiary"},
    ],

    "pedestrian_area": [
        {"highway": "pedestrian"},
        {"area:highway": "pedestrian"},
    ],

    "recreational_area": [
        {"leisure": "park"},
        {"landuse": "recreation_ground"},
    ],

    "green_area": [
        {"landuse": "grass"},
        {"natural": "grassland"},
    ],

    "forest": [
        {"natural": "wood"},
        {"landuse": "forest"},
    ],

    "waterfront": [
        {"natural": "water"},
        {"waterway": "river"},
    ],

    "rail_infrastructure": [
        {"railway": "rail"},
        {"railway": "station"},
    ],

    "construction_site": [
        {"landuse": "construction"},
        {"building": "construction"},
    ],

    "institutional_area": [
        {"office": "government"},
    ],
}


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=DEFAULT_RADIUS_METERS,
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    return parser.parse_args()


def default_output_path(input_path):

    path = Path(input_path)

    return path.with_name(
        f"semantic_validation_{path.name}"
    )


def iter_entries(data):

    if "results_by_country" in data:

        for country, entries in data[
            "results_by_country"
        ].items():

            for entry in entries:

                yield country, entry

    elif "results" in data:

        for entry in data["results"]:

            yield "unknown", entry


def build_overpass_query(
    lat,
    lon,
    radius,
):

    query_parts = []

    for key in sorted(TAG_KEYS):

        query_parts.append(
            f'node["{key}"](around:{radius},{lat},{lon});'
        )

        query_parts.append(
            f'way["{key}"](around:{radius},{lat},{lon});'
        )

        query_parts.append(
            f'relation["{key}"](around:{radius},{lat},{lon});'
        )

    body = "\n".join(query_parts)

    return f"""
    [out:json][timeout:25];

    (
        {body}
    );

    out body center;
    """


def fetch_osm_elements(
    lat,
    lon,
    radius,
    timeout,
):

    query = build_overpass_query(
        lat,
        lon,
        radius,
    )

    last_error = None

    for url in OVERPASS_URLS:

        try:

            response = requests.post(

                url,

                data={"data": query},

                timeout=timeout,

                headers={
                    "User-Agent":
                    "GeoSemanticValidator/6.0"
                },
            )

            if response.status_code != 200:

                last_error = (
                    f"HTTP "
                    f"{response.status_code}"
                )

                continue

            data = response.json()

            elements = data.get(
                "elements",
                [],
            )

            parsed = []

            for element in elements:

                tags = element.get("tags")

                if not tags:
                    continue

                parsed.append({

                    "id":
                    element.get("id"),

                    "type":
                    element.get("type"),

                    "tags":
                    tags,
                })

            return parsed

        except Exception as exc:

            last_error = str(exc)

    raise RuntimeError(
        f"All Overpass endpoints failed. "
        f"Last error: {last_error}"
    )


def normalize_feature_name(name):

    if not name:
        return "unknown"

    return (
        str(name)
        .strip()
        .lower()
    )


def normalize_osm_queries(osm_tags):

    if isinstance(osm_tags, dict):

        return [osm_tags]

    if isinstance(osm_tags, list):

        return [

            item

            for item in osm_tags

            if isinstance(item, dict)
        ]

    return []


def tag_query_matches(
    query,
    tags,
):

    for key, expected_value in query.items():

        if key not in tags:
            return False

        real_value = str(
            tags.get(key)
        ).lower()

        if real_value != str(
            expected_value
        ).lower():

            return False

    return True


def expand_feature_queries(
    feature_name,
    original_queries,
):

    semantic_queries = FEATURE_EQUIVALENCE.get(
        feature_name,
        [],
    )

    expanded = []

    expanded.extend(original_queries)
    expanded.extend(semantic_queries)

    unique = []

    seen = set()

    for item in expanded:

        frozen = tuple(
            sorted(item.items())
        )

        if frozen in seen:
            continue

        seen.add(frozen)

        unique.append(item)

    return unique


def find_matches(
    feature_item,
    osm_elements,
):

    feature_name = normalize_feature_name(
        feature_item.get("feature")
    )

    original_queries = normalize_osm_queries(
        feature_item.get("osm_tags")
    )

    expanded_queries = expand_feature_queries(
        feature_name,
        original_queries,
    )

    matches = []

    for element in osm_elements:

        tags = element.get("tags", {})

        if any(

            tag_query_matches(
                query,
                tags,
            )

            for query in expanded_queries
        ):

            matches.append({

                "id":
                element.get("id"),

                "type":
                element.get("type"),

                "tags":
                tags,
            })

    return matches


def infer_real_features(osm_elements):

    real_features = {}

    for feature_name, queries in FEATURE_EQUIVALENCE.items():

        matches = []

        for element in osm_elements:

            tags = element.get("tags", {})

            if any(

                tag_query_matches(
                    query,
                    tags,
                )

                for query in queries
            ):

                matches.append({

                    "id":
                    element.get("id"),

                    "type":
                    element.get("type"),

                    "tags":
                    tags,
                })

        if len(matches) >= MIN_MATCHES_FOR_REAL_FEATURE:

            real_features[feature_name] = {

                "matches_count":
                len(matches),

                "matched_elements_sample":
                matches[:10],
            }

    return real_features


def calculate_match_score(matches):

    if not matches:
        return 0.0

    score = 0

    for match in matches:

        tags = match.get("tags", {})

        if tags.get("landuse") == "commercial":
            score += 3

        if tags.get("building") == "commercial":
            score += 3

        if tags.get("building") == "retail":
            score += 3

        if tags.get("landuse") == "residential":
            score += 3

        if tags.get("building") == "apartments":
            score += 3

        if tags.get("landuse") == "industrial":
            score += 3

        if tags.get("highway") == "pedestrian":
            score += 3

        if tags.get("highway") in {
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
        }:
            score += 2

        if tags.get("railway") in {
            "rail",
            "station",
        }:
            score += 3

        if tags.get("leisure") == "park":
            score += 3

        if tags.get("natural") == "wood":
            score += 3

        if tags.get("landuse") == "forest":
            score += 3

        if tags.get("natural") == "water":
            score += 3

        score += 1

    normalized = min(
        score / 10.0,
        1.0,
    )

    return round(
        normalized,
        3,
    )


def validate_geographic_features(
    geographic_features,
    osm_elements,
):

    results = {}

    predicted_feature_names = set()

    for feature_item in geographic_features or []:

        if not isinstance(
            feature_item,
            dict,
        ):
            continue

        feature_name = normalize_feature_name(
            feature_item.get("feature")
        )

        predicted_feature_names.add(
            feature_name
        )

        original_queries = normalize_osm_queries(
            feature_item.get("osm_tags")
        )

        expanded_queries = expand_feature_queries(
            feature_name,
            original_queries,
        )

        matches = find_matches(
            feature_item,
            osm_elements,
        )

        results[feature_name] = {

            "predicted":
            True,

            "real":
            len(matches) > 0,

            "semantic_match_score":
            calculate_match_score(matches),

            "original_osm_tags":
            original_queries,

            "expanded_osm_tags":
            expanded_queries,

            "matches_count":
            len(matches),

            "matched_elements_sample":
            matches[:10],
        }

    real_features = infer_real_features(
        osm_elements
    )

    for feature_name, real_data in real_features.items():

        if feature_name not in predicted_feature_names:

            results[feature_name] = {

                "predicted":
                False,

                "real":
                True,

                "semantic_match_score":
                1.0,

                "original_osm_tags":
                [],

                "expanded_osm_tags":
                FEATURE_EQUIVALENCE.get(
                    feature_name,
                    [],
                ),

                "matches_count":
                real_data["matches_count"],

                "matched_elements_sample":
                real_data[
                    "matched_elements_sample"
                ],
            }

    return results


def empty_counts():

    return {

        "tp": 0,
        "fp": 0,
        "fn": 0,
        "validated_addresses": 0,
    }


def add_counts(
    counts,
    collection,
):

    for item in collection.values():

        predicted = bool(
            item.get("predicted")
        )

        real = bool(
            item.get("real")
        )

        if predicted and real:

            counts["tp"] += 1

        elif predicted and not real:

            counts["fp"] += 1

        elif not predicted and real:

            counts["fn"] += 1


def evaluate_counts(counts):

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]

    precision = (

        tp / (tp + fp)

        if (tp + fp)

        else 0
    )

    recall = (

        tp / (tp + fn)

        if (tp + fn)

        else 0
    )

    f1 = (

        2 * precision * recall
        / (precision + recall)

        if (precision + recall)

        else 0
    )

    hallucination_rate = (

        fp / (tp + fp)

        if (tp + fp)

        else 0
    )

    return {

        "precision":
        round(precision, 3),

        "recall":
        round(recall, 3),

        "f1_score":
        round(f1, 3),

        "hallucination_rate":
        round(hallucination_rate, 3),

        "true_positive":
        tp,

        "false_positive":
        fp,

        "false_negative":
        fn,

        "predicted_total":
        tp + fp,

        "real_total":
        tp + fn,

        "validated_addresses":
        counts[
            "validated_addresses"
        ],
    }


def build_summary_by_country(validation):

    summary = {}

    for country, entries in validation.get(
        "results_by_country",
        {},
    ).items():

        feature_counts = empty_counts()

        for entry in entries:

            payload = entry.get(
                "validation",
                {},
            )

            if "error" in payload:
                continue

            feature_counts[
                "validated_addresses"
            ] += 1

            add_counts(
                feature_counts,
                payload.get(
                    "geographic_features",
                    {},
                ),
            )

        summary[country] = {

            "geographic_features":
            evaluate_counts(
                feature_counts
            )
        }

    return summary


def validate_entry(
    entry,
    radius,
    timeout,
):

    prediction = entry.get("prediction")

    if not isinstance(
        prediction,
        dict,
    ):

        return {
            "error":
            entry.get(
                "error",
                "Missing prediction.",
            )
        }

    lat = entry.get("latitude")
    lon = entry.get("longitude")

    if lat is None or lon is None:

        return {
            "error":
            "Missing latitude or longitude."
        }

    try:

        osm_elements = fetch_osm_elements(

            float(lat),
            float(lon),
            radius,
            timeout,
        )

    except Exception as exc:

        return {
            "error":
            f"OSM fetch failed: {str(exc)}"
        }

    return {

        "osm_elements_with_tags":
        len(osm_elements),

        "geographic_features":
        validate_geographic_features(

            prediction.get(
                "geographic_features",
                [],
            ),

            osm_elements,
        ),
    }


def main():

    args = parse_args()

    input_path = Path(args.input)

    output_path = (

        Path(args.output)

        if args.output

        else default_output_path(
            input_path
        )
    )

    data = json.loads(

        input_path.read_text(
            encoding="utf-8"
        )
    )

    validation = {

        "_metadata": {

            "source_file":
            str(input_path),

            "radius_meters":
            args.radius,

            "delay_seconds":
            args.delay,

            "method":
            "semantic_geographic_validation",

            "source_metadata":
            data.get(
                "_metadata",
                {},
            ),
        },

        "results_by_country": {},

        "summary_by_country": {},
    }

    validated_count = 0

    for country, entry in iter_entries(data):

        if (

            args.limit is not None

            and validated_count >= args.limit
        ):
            break

        address = entry.get(
            "address",
            "",
        )

        print(
            f"[{validated_count + 1}] "
            f"{country} - {address}"
        )

        validated_entry = {

            "city":
            entry.get("city"),

            "address":
            address,

            "latitude":
            entry.get("latitude"),

            "longitude":
            entry.get("longitude"),

            "validation":
            validate_entry(

                entry,
                args.radius,
                args.timeout,
            ),
        }

        validation[
            "results_by_country"
        ].setdefault(
            country,
            [],
        ).append(
            validated_entry
        )

        validation[
            "summary_by_country"
        ] = build_summary_by_country(
            validation
        )

        validated_count += 1

        output_path.write_text(

            json.dumps(
                validation,
                indent=2,
                ensure_ascii=False,
            ),

            encoding="utf-8",
        )

        if args.delay > 0:

            time.sleep(args.delay)

    print("\nValidation finished.")

    print(
        f"Validated addresses: "
        f"{validated_count}"
    )

    print(
        f"Saved at: "
        f"{output_path}"
    )


if __name__ == "__main__":

    main()