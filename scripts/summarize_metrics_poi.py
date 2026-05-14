import json
import pandas as pd
from pathlib import Path
from statistics import median

# =========================
# ARQUIVOS
# =========================

json_files = [

"validation/POIs/validation_sonnet_4.6_poi.json",
"validation/POIs/validation_gpt_4.1_poi.json",
"validation/POIs/validation_gemini_3flash_preview_poi.json",
"validation/POIs/validation_gemini_3.1_flash_lite_poi.json",
"validation/POIs/validation_deepseek_v4_pro_poi.json",
"validation/POIs/validation_deepseek_v4_flash_poi.json"
]

# =========================
# RESULTADOS
# =========================

country_rows = []
city_rows = []

# =========================
# FUNÇÃO AUXILIAR
# =========================

def calculate_metrics(pois):

    total = len(pois)

    if total == 0:
        return {
            "total_pois": 0,
            "name_accuracy": 0,
            "within_2km_accuracy": 0,
            "direction_accuracy": 0,
            "mae_km": 0,
            "median_error_km": 0,
            "low_confidence_rate": 0
        }

    name_matches = 0
    within_2km = 0
    direction_matches = 0
    low_confidence = 0

    distance_errors = []

    for poi in pois:

        validation = poi.get("validation", {})

        # =========================
        # LOW CONFIDENCE
        # =========================

        if poi.get("status") == "low_confidence_match":
            low_confidence += 1

        # =========================
        # NAME MATCH
        # =========================

        if validation.get("name_match") is True:
            name_matches += 1

        # =========================
        # WITHIN 2KM
        # =========================

        if validation.get("within_2km") is True:
            within_2km += 1

        # =========================
        # DIRECTION MATCH
        # =========================

        if validation.get("direction_match") is True:
            direction_matches += 1

        # =========================
        # DISTANCE ERROR
        # =========================

        distance_diff = validation.get(
            "distance_difference_km"
        )

        if distance_diff is not None:
            distance_errors.append(distance_diff)

    # =========================
    # MÉTRICAS
    # =========================

    mae_km = (
        sum(distance_errors) / len(distance_errors)
        if distance_errors else 0
    )

    median_error_km = (
        median(distance_errors)
        if distance_errors else 0
    )

    return {

        "total_pois": total,

        "name_accuracy":
            round(name_matches / total, 4),

        "within_2km_accuracy":
            round(within_2km / total, 4),

        "direction_accuracy":
            round(direction_matches / total, 4),

        "mae_km":
            round(mae_km, 4),

        "median_error_km":
            round(median_error_km, 4),

        "low_confidence_rate":
            round(low_confidence / total, 4)
    }

# =========================
# PROCESSAMENTO
# =========================

for file_path in json_files:

    print(f"\nProcessando: {file_path}")

    try:

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    except Exception as e:
        print(f"Erro ao abrir JSON: {e}")
        continue

    # =========================
    # METADADOS
    # =========================

    model_name = (
        data.get("_metadata", {})
            .get("model", Path(file_path).stem)
    )

    results = data.get("results", [])

    # =========================
    # AGRUPAMENTOS
    # =========================

    country_groups = {}
    city_groups = {}

    # =========================
    # PROCESSAR ENDEREÇOS
    # =========================

    for item in results:

        country = item.get("country", "Unknown")
        city = item.get("city", "Unknown")

        validated_pois = item.get(
            "validated_pois",
            []
        )

        # =========================
        # PAÍS
        # =========================

        if country not in country_groups:
            country_groups[country] = []

        country_groups[country].extend(validated_pois)

        # =========================
        # CIDADE
        # =========================

        city_key = (country, city)

        if city_key not in city_groups:
            city_groups[city_key] = []

        city_groups[city_key].extend(validated_pois)

    # =========================
    # MÉTRICAS POR PAÍS
    # =========================

    for country, pois in country_groups.items():

        metrics = calculate_metrics(pois)

        country_rows.append({
            "model": model_name,
            "country": country,
            **metrics
        })

    # =========================
    # MÉTRICAS POR CIDADE
    # =========================

    for (country, city), pois in city_groups.items():

        metrics = calculate_metrics(pois)

        city_rows.append({
            "model": model_name,
            "country": country,
            "city": city,
            **metrics
        })

# =========================
# DATAFRAMES
# =========================

df_country = pd.DataFrame(country_rows)
df_city = pd.DataFrame(city_rows)

# =========================
# ORDENAR
# =========================

if not df_country.empty:

    df_country = df_country.sort_values(
        ["country", "within_2km_accuracy"],
        ascending=[True, False]
    )

if not df_city.empty:

    df_city = df_city.sort_values(
        ["country", "city", "within_2km_accuracy"],
        ascending=[True, True, False]
    )

# =========================
# SALVAR CSV
# =========================

df_country.to_csv(
    "csv/poi_metrics_by_country.csv",
    index=False,
    encoding="utf-8"
)

df_city.to_csv(
    "csv/poi_metrics_by_city.csv",
    index=False,
    encoding="utf-8"
)

# =========================
# EXIBIR
# =========================

print("\n===== MÉTRICAS POR PAÍS =====")
print(df_country)

print("\n===== MÉTRICAS POR CIDADE =====")
print(df_city)

print("\nArquivos gerados:")
print("- poi_metrics_by_country.csv")
print("- poi_metrics_by_city.csv")