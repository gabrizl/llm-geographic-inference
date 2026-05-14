import json
import pandas as pd
from pathlib import Path

# =========================
# ARQUIVOS DOS MODELOS
# =========================

json_files = [
    "validation/geographic_features/semantic_validation_geographic_features_claude-sonnet-4-6.json",
    "validation/geographic_features/semantic_validation_geographic_features_deepseek-v4-flash.json",
    "validation/geographic_features/semantic_validation_geographic_features_deepseek-v4-pro.json",
    "validation/geographic_features/semantic_validation_geographic_features_gemini-3-1-flash-lite.json",
    "validation/geographic_features/semantic_validation_geographic_features_gemini-3-flash-preview.json",
    "validation/geographic_features/semantic_validation_geographic_features_openai_gpt-4-1.json"
]

# =========================
# FUNÇÕES
# =========================

def calculate_metrics(tp, fp, fn):

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    hallucination_rate = (
        fp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "hallucination_rate": round(hallucination_rate, 3)
    }


# =========================
# RESULTADOS
# =========================

country_rows = []
city_rows = []

# =========================
# PROCESSAMENTO
# =========================

for file_path in json_files:

    print(f"\nProcessando: {file_path}")

    try:

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    except Exception as e:
        print(f"Erro ao abrir arquivo: {e}")
        continue

    # =========================
    # METADADOS
    # =========================

    model_name = (
        data.get("_metadata", {})
            .get("source_metadata", {})
            .get("model", Path(file_path).stem)
    )

    results_by_country = data.get("results_by_country", {})

    # ==========================================
    # PROCESSAR PAÍSES
    # ==========================================

    for country, addresses in results_by_country.items():

        country_tp = 0
        country_fp = 0
        country_fn = 0

        city_stats = {}

        # ==========================================
        # PROCESSAR ENDEREÇOS
        # ==========================================

        for item in addresses:

            city = item.get("city", "Unknown")

            if city not in city_stats:
                city_stats[city] = {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0
                }

            # ==========================================
            # VALIDATION
            # ==========================================

            validation = item.get("validation", {})

            geographic_features = validation.get(
                "geographic_features",
                {}
            )

            # Se não existir geographic_features
            if not geographic_features:
                continue

            # ==========================================
            # PROCESSAR FEATURES
            # ==========================================

            for feature_name, feature_data in geographic_features.items():

                predicted = feature_data.get("predicted", False)
                real = feature_data.get("real", False)

                # ==========================
                # TRUE POSITIVE
                # ==========================

                if predicted and real:

                    country_tp += 1
                    city_stats[city]["tp"] += 1

                # ==========================
                # FALSE POSITIVE
                # ==========================

                elif predicted and not real:

                    country_fp += 1
                    city_stats[city]["fp"] += 1

                # ==========================
                # FALSE NEGATIVE
                # ==========================

                elif not predicted and real:

                    country_fn += 1
                    city_stats[city]["fn"] += 1

                # TN ignorado
                # not predicted and not real

        # ==========================================
        # CALCULAR MÉTRICAS DO PAÍS
        # ==========================================

        metrics = calculate_metrics(
            country_tp,
            country_fp,
            country_fn
        )

        country_rows.append({
            "model": model_name,
            "country": country,
            "true_positive": country_tp,
            "false_positive": country_fp,
            "false_negative": country_fn,
            **metrics
        })

        # ==========================================
        # CALCULAR MÉTRICAS DAS CIDADES
        # ==========================================

        for city, stats in city_stats.items():

            city_metrics = calculate_metrics(
                stats["tp"],
                stats["fp"],
                stats["fn"]
            )

            city_rows.append({
                "model": model_name,
                "country": country,
                "city": city,
                "true_positive": stats["tp"],
                "false_positive": stats["fp"],
                "false_negative": stats["fn"],
                **city_metrics
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
        ["country", "f1_score"],
        ascending=[True, False]
    )

if not df_city.empty:

    df_city = df_city.sort_values(
        ["country", "city", "f1_score"],
        ascending=[True, True, False]
    )

# =========================
# SALVAR CSV
# =========================

df_country.to_csv(
    "csv/geographic_features_metrics_by_country.csv",
    index=False,
    encoding="utf-8"
)

df_city.to_csv(
    "csv/geographic_features_metrics_by_city.csv",
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
print("- metrics_by_country.csv")
print("- metrics_by_city.csv")