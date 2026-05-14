import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from statistics import mean, median

import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz

load_dotenv()

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_MAPS_API_KEY"
)

if not GOOGLE_API_KEY:

    raise RuntimeError(
        "Missing GOOGLE_MAPS_API_KEY"
    )

# =========================================================
# CONFIG
# =========================================================

SEARCH_RADIUS_METERS = 10000

MIN_NAME_SIMILARITY = 85

MIN_CANDIDATE_SCORE = 60

TEXT_SEARCH_URL = (
    "https://maps.googleapis.com/maps/api/place/textsearch/json"
)

DISTANCE_MATRIX_URL = (
    "https://maps.googleapis.com/maps/api/distancematrix/json"
)

CACHE_DIR = Path("cache")

TEXT_CACHE_DIR = (
    CACHE_DIR / "text_search"
)

DISTANCE_CACHE_DIR = (
    CACHE_DIR / "distance_matrix"
)

TEXT_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DISTANCE_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# =========================================================
# GENERIC WORDS
# =========================================================

GENERIC_WORDS = {

    "escola",
    "municipal",
    "estadual",
    "supermercado",
    "mercado",
    "restaurante",
    "unidade",
    "basica",
    "saude",
    "upa",
    "ubs",
    "hospital",
    "posto",
    "padaria",
    "farmacia",
    "shopping",
    "centro",
    "ltda",
    "me",
    "eireli",
    "store",
}

# =========================================================
# TYPE MAPPING
# =========================================================

GOOGLE_TYPE_MAPPING = {

    "restaurant": "restaurant",
    "cafe": "restaurant",
    "bakery": "restaurant",
    "meal_takeaway": "restaurant",
    "food": "restaurant",

    "school": "school",
    "university": "school",
    "primary_school": "school",
    "secondary_school": "school",

    "hospital": "hospital",
    "doctor": "hospital",
    "clinic": "hospital",
    "health": "hospital",

    "park": "park",
    "campground": "park",
    "garden": "park",

    "store": "retail",
    "shopping_mall": "retail",
    "supermarket": "retail",
    "pharmacy": "retail",
    "convenience_store": "retail",

    "subway_station": "transport",
    "train_station": "transport",
    "transit_station": "transport",
    "bus_station": "transport",
    "taxi_stand": "transport",
    "airport": "transport",

    "tourist_attraction": "landmark",
    "museum": "landmark",
    "stadium": "landmark",
    "city_hall": "landmark",

    "church": "religious",
    "mosque": "religious",
    "synagogue": "religious",
    "hindu_temple": "religious",
    "place_of_worship": "religious",

    "apartment_complex": "residential",

    "industrial_estate": "industrial",
    "warehouse": "industrial",
}

# =========================================================
# ARGUMENTS
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Validate POIs "
            "using Google Maps API"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--provider",
        default="unknown",
    )

    parser.add_argument(
        "--model",
        default="unknown",
    )

    return parser.parse_args()

# =========================================================
# CACHE
# =========================================================

def make_cache_key(*parts):

    raw = "||".join(
        map(str, parts)
    )

    return hashlib.md5(
        raw.encode("utf-8")
    ).hexdigest()


def load_cache(path):

    if path.exists():

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    return None


def save_cache(path, data):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
        )

# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


def normalize_name(text):

    text = normalize_text(text)

    for word in GENERIC_WORDS:

        text = re.sub(
            rf"\b{word}\b",
            "",
            text,
        )

    return " ".join(
        text.split()
    )


def extract_important_tokens(text):

    normalized = normalize_name(text)

    tokens = set()

    for token in normalized.split():

        if len(token) >= 3:
            tokens.add(token)

    return tokens


def normalize_type(google_types):

    for gtype in google_types:

        mapped = GOOGLE_TYPE_MAPPING.get(
            gtype
        )

        if mapped:
            return mapped

    return None

# =========================================================
# GOOGLE SEARCH
# =========================================================

def text_search(
    query,
    lat,
    lon,
):

    cache_key = make_cache_key(
        query,
        round(lat, 5),
        round(lon, 5),
    )

    cache_path = (
        TEXT_CACHE_DIR
        / f"{cache_key}.json"
    )

    cached = load_cache(
        cache_path
    )

    if cached is not None:
        return cached

    params = {

        "query":
        query,

        "location":
        f"{lat},{lon}",

        "radius":
        SEARCH_RADIUS_METERS,

        "key":
        GOOGLE_API_KEY,
    }

    response = requests.get(
        TEXT_SEARCH_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    results = data.get(
        "results",
        [],
    )

    if not results:

        print(
            f"[NOT FOUND] {query}"
        )

    else:

        print(
            f"[FOUND] {query} "
            f"({len(results)} candidates)"
        )

        save_cache(
            cache_path,
            results,
        )

    time.sleep(0.1)

    return results

# =========================================================
# DISTANCE MATRIX
# =========================================================

def get_walking_distance_km(
    origin_lat,
    origin_lon,
    dest_lat,
    dest_lon,
):

    cache_key = make_cache_key(

        round(origin_lat, 5),
        round(origin_lon, 5),

        round(dest_lat, 5),
        round(dest_lon, 5),
    )

    cache_path = (

        DISTANCE_CACHE_DIR
        / f"{cache_key}.json"
    )

    cached = load_cache(
        cache_path
    )

    if cached is not None:
        return cached

    params = {

        "origins":
        f"{origin_lat},{origin_lon}",

        "destinations":
        f"{dest_lat},{dest_lon}",

        "mode":
        "walking",

        "key":
        GOOGLE_API_KEY,
    }

    response = requests.get(
        DISTANCE_MATRIX_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    rows = data.get(
        "rows",
        [],
    )

    if not rows:
        return None

    elements = rows[0].get(
        "elements",
        [],
    )

    if not elements:
        return None

    distance = (

        elements[0]
        .get("distance", {})
        .get("value")
    )

    if distance is None:
        return None

    km = round(
        distance / 1000,
        2,
    )

    save_cache(
        cache_path,
        km,
    )

    time.sleep(0.1)

    return km

# =========================================================
# DIRECTION
# =========================================================

def calculate_bearing(
    lat1,
    lon1,
    lat2,
    lon2,
):

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    diff_lon = math.radians(
        lon2 - lon1
    )

    x = (
        math.sin(diff_lon)
        * math.cos(lat2)
    )

    y = (
        math.cos(lat1)
        * math.sin(lat2)
        - math.sin(lat1)
        * math.cos(lat2)
        * math.cos(diff_lon)
    )

    initial_bearing = math.atan2(
        x,
        y,
    )

    initial_bearing = math.degrees(
        initial_bearing
    )

    return (
        initial_bearing + 360
    ) % 360


def bearing_to_direction(bearing):

    directions = [
        "N",
        "NE",
        "E",
        "SE",
        "S",
        "SW",
        "W",
        "NW",
    ]

    index = round(
        bearing / 45
    ) % 8

    return directions[index]


def direction_match(
    predicted,
    actual,
):

    if predicted == actual:
        return True

    adjacency = {

        "N": {"NE", "NW"},
        "NE": {"N", "E"},
        "E": {"NE", "SE"},
        "SE": {"E", "S"},
        "S": {"SE", "SW"},
        "SW": {"S", "W"},
        "W": {"SW", "NW"},
        "NW": {"N", "W"},
    }

    return (
        actual in adjacency.get(
            predicted,
            set(),
        )
    )

# =========================================================
# MATCHING
# =========================================================

def exact_match(
    poi_name,
    candidate_name,
):

    return (
        normalize_text(poi_name)
        ==
        normalize_text(candidate_name)
    )


def containment_match(
    poi_name,
    candidate_name,
):

    a = normalize_text(poi_name)
    b = normalize_text(candidate_name)

    return (
        a in b
        or
        b in a
    )


def important_token_overlap(
    poi_name,
    candidate_name,
):

    poi_tokens = extract_important_tokens(
        poi_name
    )

    candidate_tokens = (
        extract_important_tokens(
            candidate_name
        )
    )

    if not poi_tokens:
        return 0

    overlap = (
        poi_tokens
        &
        candidate_tokens
    )

    return (
        len(overlap)
        / len(poi_tokens)
    )

# =========================================================
# CANDIDATE SCORING
# =========================================================

def compute_candidate_score(
    poi,
    candidate,
    reference_lat,
    reference_lon,
):

    candidate_name = candidate.get(
        "name",
        ""
    )

    google_types = candidate.get(
        "types",
        []
    )

    google_type = normalize_type(
        google_types
    )

    type_match = (
        poi["type"]
        ==
        google_type
    )

    if exact_match(
        poi["name"],
        candidate_name,
    ):

        base_similarity = 100

    elif containment_match(
        poi["name"],
        candidate_name,
    ):

        base_similarity = 95

    else:

        overlap = important_token_overlap(

            poi["name"],
            candidate_name,
        )

        if overlap == 0:
            return -999

        fuzzy_score = fuzz.token_set_ratio(

            normalize_name(
                poi["name"]
            ),

            normalize_name(
                candidate_name
            ),
        )

        base_similarity = (
            fuzzy_score * overlap
        )

    score = base_similarity

    if type_match:
        score += 25
    else:
        score -= 40

    google_lat = (
        candidate["geometry"]
        ["location"]["lat"]
    )

    google_lon = (
        candidate["geometry"]
        ["location"]["lng"]
    )

    walking_distance = (
        get_walking_distance_km(
            reference_lat,
            reference_lon,
            google_lat,
            google_lon,
        )
    )

    if walking_distance is not None:

        predicted_distance = (
            poi["distance_km"]
        )

        distance_difference = abs(

            predicted_distance
            - walking_distance
        )

        score -= (
            distance_difference * 15
        )

        if walking_distance > 2.5:
            score -= 30

    return round(score, 2)

# =========================================================
# VALIDATION
# =========================================================

def validate_poi(
    reference_lat,
    reference_lon,
    city,
    address,
    poi,
):

    queries = [

        (
            f"{poi['name']}, "
            f"{address}, "
            f"{city}"
        ),

        (
            f"{poi['name']} "
            f"{city}"
        ),

        poi["name"],
    ]

    results = []

    for query in queries:

        results = text_search(
            query,
            reference_lat,
            reference_lon,
        )

        if results:
            break

    if not results:

        return {
            "status": "not_found"
        }

    scored_results = []

    for candidate in results:

        score = compute_candidate_score(

            poi=poi,

            candidate=candidate,

            reference_lat=reference_lat,

            reference_lon=reference_lon,
        )

        scored_results.append(
            (score, candidate)
        )

    scored_results.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    best_score, best = (
        scored_results[0]
    )

    if best_score < MIN_CANDIDATE_SCORE:

        return {

            "status":
            "low_confidence_match",

            "predicted":
            poi,

            "validation": {

                "candidate_score":
                best_score,
            }
        }

    print(
        f"[MATCH SCORE {best_score}] "
        f"{poi['name']} -> "
        f"{best.get('name')}"
    )

    google_name = best.get(
        "name",
        ""
    )

    google_lat = (
        best["geometry"]
        ["location"]["lat"]
    )

    google_lon = (
        best["geometry"]
        ["location"]["lng"]
    )

    similarity = fuzz.token_set_ratio(

        normalize_name(
            poi["name"]
        ),

        normalize_name(
            google_name
        ),
    )

    walking_distance = (
        get_walking_distance_km(
            reference_lat,
            reference_lon,
            google_lat,
            google_lon,
        )
    )

    distance_difference = None

    within_2km = False

    if walking_distance is not None:

        distance_difference = round(

            abs(
                poi["distance_km"]
                - walking_distance
            ),

            2,
        )

        within_2km = (
            walking_distance <= 2.0
        )

    bearing = calculate_bearing(

        reference_lat,
        reference_lon,

        google_lat,
        google_lon,
    )

    actual_direction = (
        bearing_to_direction(
            bearing
        )
    )

    predicted_direction = (
        poi["direction"]
    )

    direction_ok = direction_match(
        predicted_direction,
        actual_direction,
    )

    name_match = (
        similarity
        >= MIN_NAME_SIMILARITY
    )

    return {

        "predicted": poi,

        "google": {

            "name":
            google_name,

            "latitude":
            google_lat,

            "longitude":
            google_lon,

            "walking_distance_km":
            walking_distance,

            "direction":
            actual_direction,
        },

        "validation": {

            "candidate_score":
            best_score,

            # =============================
            # POI GROUNDING
            # =============================

            "name_similarity":
            similarity,

            "name_match":
            name_match,

            # =============================
            # DISTANCE
            # =============================

            "distance_difference_km":
            distance_difference,

            "within_2km":
            within_2km,

            # =============================
            # DIRECTION
            # =============================

            "direction_match":
            direction_ok,
        },
    }

# =========================================================
# METRICS
# =========================================================

def init_stats():

    return {

        "total_pois": 0,

        "name_matches": 0,

        "within_2km_matches": 0,

        "direction_matches": 0,

        "distance_errors": [],

        "not_found": 0,

        "low_confidence": 0,
    }


def update_stats(stats, poi):

    stats["total_pois"] += 1

    status = poi.get("status")

    if status == "not_found":

        stats["not_found"] += 1
        return

    if status == "low_confidence_match":

        stats["low_confidence"] += 1
        return

    validation = poi.get(
        "validation",
        {}
    )

    if validation.get(
        "name_match"
    ):
        stats["name_matches"] += 1

    if validation.get(
        "within_2km"
    ):
        stats["within_2km_matches"] += 1

    if validation.get(
        "direction_match"
    ):
        stats["direction_matches"] += 1

    distance_error = validation.get(
        "distance_difference_km"
    )

    if distance_error is not None:

        stats[
            "distance_errors"
        ].append(
            distance_error
        )


def finalize_stats(stats):

    total = stats["total_pois"]

    distance_errors = stats[
        "distance_errors"
    ]

    def safe_div(a, b):

        return round(a / b, 4) if b else 0

    return {

        "total_pois":
        total,

        # =============================
        # POI GROUNDING
        # =============================

        "name_accuracy":
        safe_div(
            stats["name_matches"],
            total,
        ),

        # =============================
        # SPATIAL CONSTRAINT
        # =============================

        "within_2km_accuracy":
        safe_div(
            stats["within_2km_matches"],
            total,
        ),

        # =============================
        # DIRECTIONAL REASONING
        # =============================

        "direction_accuracy":
        safe_div(
            stats["direction_matches"],
            total,
        ),

        # =============================
        # DISTANCE ESTIMATION
        # =============================

        "mae_km":
        round(
            mean(distance_errors),
            3,
        ) if distance_errors else None,

        "median_error_km":
        round(
            median(distance_errors),
            3,
        ) if distance_errors else None,

        # =============================
        # OTHER
        # =============================

        "not_found_rate":
        safe_div(
            stats["not_found"],
            total,
        ),

        "low_confidence_rate":
        safe_div(
            stats["low_confidence"],
            total,
        ),
    }


def compute_aggregate_metrics(
    validated_results,
):

    overall_stats = init_stats()

    by_country = {}

    by_city = {}

    for entry in validated_results:

        country = entry.get(
            "country",
            "unknown"
        )

        city = entry.get(
            "city",
            "unknown"
        )

        country_stats = (
            by_country.setdefault(
                country,
                init_stats()
            )
        )

        city_stats = (
            by_city.setdefault(
                city,
                init_stats()
            )
        )

        for poi in entry.get(
            "validated_pois",
            [],
        ):

            update_stats(
                overall_stats,
                poi
            )

            update_stats(
                country_stats,
                poi
            )

            update_stats(
                city_stats,
                poi
            )

    finalized_cities = {

        city: finalize_stats(stats)

        for city, stats
        in by_city.items()
    }

    finalized_cities = dict(

        sorted(

            finalized_cities.items(),

            key=lambda item:
            item[1][
                "name_accuracy"
            ],

            reverse=True,
        )
    )

    return {

        "overall":
        finalize_stats(
            overall_stats
        ),

        "by_country": {

            country:
            finalize_stats(stats)

            for country, stats
            in by_country.items()
        },

        "by_city":
        finalized_cities,
    }

# =========================================================
# MAIN
# =========================================================

def main():

    args = parse_args()

    with open(
        args.input,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    validated = {

        "_metadata": {

            "validator":
            "google_maps_api",

            "provider":
            args.provider,

            "model":
            args.model,
        },

        "results": [],
    }

    entries = data["results"]

    total = len(entries)

    for index, entry in enumerate(
        entries,
        start=1,
    ):

        print(
            f"[{index}/{total}] "
            f"{entry['city']} - "
            f"{entry['address']}"
        )

        if "pois" not in entry:

            validated[
                "results"
            ].append({
                **entry
            })

            continue

        ref_lat = entry["latitude"]
        ref_lon = entry["longitude"]

        validated_pois = []

        for poi in entry["pois"]:

            try:

                result = validate_poi(

                    reference_lat=ref_lat,

                    reference_lon=ref_lon,

                    city=entry["city"],

                    address=entry["address"],

                    poi=poi,
                )

                validated_pois.append(
                    result
                )

            except Exception as exc:

                validated_pois.append({

                    "predicted":
                    poi,

                    "error":
                    str(exc),
                })

        validated["results"].append({

            "country":
            entry.get(
                "country",
                "unknown"
            ),

            "city":
            entry["city"],

            "address":
            entry["address"],

            "latitude":
            ref_lat,

            "longitude":
            ref_lon,

            "validated_pois":
            validated_pois,
        })

    validated["summary"] = (
        compute_aggregate_metrics(
            validated["results"]
        )
    )

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            validated,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nSaved to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()