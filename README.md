# Spatial LLM Benchmark: Geographic Inference with Large Language Models

This repository contains the dataset, source code, and evaluation results for the short paper:

**"Geographic Inference with Large Language Models: Identification of POIs and Urban Features from Addresses and Coordinates"** (Brazilian Symposium on Databases - SBBD 2026)

---

## Abstract

This paper evaluates the ability of Large Language Models (LLMs) to infer geographic features and identify points of interest (POIs) from addresses and coordinates. Nine cities of different urban sizes, distributed across three continents, were analyzed, with five addresses per city in scenarios such as urban center, residential area, green area, proximity to water and mixed area. The predictions were validated with external geospatial databases. The results indicate greater stability in feature inference than in POI identification, whose performance varies according to country and urban size; Gemini 3.1 Flash Lite leads in large and medium-sized cities, with a tie between Gemini 3 Flash Preview and DeepSeek V4 Flash in small cities.

---

## Repository Structure

```text
├── csv/
│   ├── addresses.csv                               # Core dataset of 45 evaluation addresses
│   ├── geographic_features_metrics_by_city.csv     # Calculated metrics by city (features)
│   ├── geographic_features_metrics_by_country.csv  # Calculated metrics by country (features)
│   ├── poi_metrics_by_city.csv                     # Calculated metrics by city (POIs)
│   └── poi_metrics_by_country.csv                  # Calculated metrics by country (POIs)
├── prompts/
│   ├── address_user_prompt.txt                     # Prompt used for urban feature inference
│   └── POI_list_prompt.txt                         # Prompt used for POI identification
├── results/
│   ├── geographic_features/                        # Raw predictions for urban features
│   └── POIs/                                       # Raw predictions for POIs (standardized naming)
├── scripts/                                        # Execution and evaluation scripts
│   ├── run_*_geographic_features.py                # Model inference runner for features
│   ├── run_*_poi_list.py                           # Model inference runner for POIs
│   ├── validate_geographic_features.py             # Validation script using Overpass (OSM)
│   ├── validate_poi_list.py                        # Validation script using Google Places API
│   ├── summarize_metrics_geographic_features.py    # Metric calculator for features
│   └── summarize_metrics_poi.py                    # Metric calculator for POIs
├── src/                                            # Shared utility scripts
│   ├── env_loader.py
│   └── prompt_loader.py
├── tests/                                          # Unit tests
└── validation/
    ├── geographic_features/                        # Semantic validation outputs for features
    └── POIs/                                       # Validation outputs for POIs (standardized naming)
```

---

## Dataset & Methodology

### 1. Cities and Scenarios
The benchmark spans **9 cities** across **3 continents**, categorized by urban scale:
- **Large Cities**: New York (USA), São Paulo (Brazil), London (UK)
- **Medium Cities**: Vancouver (Canada), Campina Grande (Brazil), Amsterdam (Netherlands)
- **Small Cities**: Aspen (USA), Tibau do Sul (Brazil), Bath (UK)

For each city, **5 distinct addresses** were selected corresponding to specific geographic scenarios:
1. **Urban Center** (high-density retail and business)
2. **Residential Area** (housing and neighborhoods)
3. **Green Area** (parks and natural conservation)
4. **Proximity to Water** (rivers, lakes, or beachfronts)
5. **Mixed Area** (commercial and residential integration)

### 2. Evaluation Tasks
- **Urban Feature Inference**: Evaluates the LLM's capacity to infer ambient urban features (e.g., commercial area, residential area, green area) within a 200m buffer of the address. Checked against **OpenStreetMap (OSM)** data using the **Overpass API**.
- **POI Identification**: Evaluates the LLM's capability to pinpoint 10 concrete Points of Interest (POIs) within a 2km radius. Checked against **Google Places API** to assess accuracy in names, distance, and relative bearing.

---

## Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/gabrizl/llm-geographic-inference.git
   cd llm-geographic-inference
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # On Windows
   source venv/bin/activate    # On Unix/macOS
   pip install -r requirements.txt
   ```
3. Set up environment variables by copying `.env.example` to `.env` and adding your API keys:
   ```env
   OPENAI_API_KEY=your-key
   GEMINI_API_KEY=your-key
   DEEPSEEK_API_KEY=your-key
   ANTHROPIC_API_KEY=your-key
   GOOGLE_MAPS_API_KEY=your-key   # Required for POI validation
   ```

---

## How to Run

All executions are modularized under the `scripts/` directory:

### 1. Model Predictions (Inference)
To generate predictions, execute the specific provider runner. E.g., to run POI identification for Gemini:
```powershell
python scripts/run_gemini_poi_list.py --input csv/addresses.csv --output results/POIs/poi_results_gemini-3-flash-preview.json
```
For geographic features inference:
```powershell
python scripts/run_openai_geographic_features.py --input csv/addresses.csv --output results/geographic_features/geographic_features_openai_gpt-4-1.json
```

### 2. Ground-Truth Validation
To validate predictions against geospatial databases:
- **For Geographic Features** (runs Overpass API queries to OpenStreetMap):
  ```powershell
  python scripts/validate_geographic_features.py --input results/geographic_features/geographic_features_deepseek-v4-flash.json
  ```
- **For POIs** (queries Google Places API):
  ```powershell
  python scripts/validate_poi_list.py --input results/POIs/poi_results_deepseek-v4-flash.json --output validation/POIs/validation_poi_deepseek-v4-flash.json --provider deepseek --model deepseek-v4-flash
  ```

### 3. Metric Summarization
Generate the aggregate CSV metrics under the `csv/` folder:
- **For Geographic Features**:
  ```powershell
  python scripts/summarize_metrics_geographic_features.py
  ```
- **For POIs**:
  ```powershell
  python scripts/summarize_metrics_poi.py
  ```

---

## Evaluation Results Summary

Below is an overview of the benchmark results aggregated by country.

### 1. POI Identification Accuracy (Within 2km)

| Model | Brazil | Canada | United States | United Kingdom | Netherlands |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemini 3 Flash Preview** | **33.3%** | **56.1%** | 47.6% | **72.6%** | **50.0%** |
| **GPT-4.1 (OpenAI)** | 30.0% | 56.0% | 48.0% | 50.0% | 40.0% |
| **DeepSeek V4 Pro** | 19.3% | 54.0% | **59.0%** | 58.0% | 40.0% |
| **Gemini 3.1 Flash Lite** | 27.3% | 48.0% | 49.0% | 54.0% | 42.0% |
| **Claude 3.5 Sonnet (4.6)** | 20.0% | 45.0% | 39.0% | 50.0% | 40.0% |
| **DeepSeek V4 Flash** | 26.7% | 42.0% | 54.0% | 58.0% | 34.0% |

### 2. Geographic Features F1-Scores

| Model | Brazil | Canada | United States | United Kingdom | Netherlands |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemini 3 Flash Preview** | **0.644** | 0.791 | **0.716** | 0.767 | 0.588 |
| **Claude 3.5 Sonnet (4.6)** | 0.559 | 0.625 | 0.702 | **0.769** | **0.621** |
| **Gemini 3.1 Flash Lite** | 0.554 | **0.791** | 0.658 | 0.725 | 0.612 |
| **DeepSeek V4 Flash** | 0.551 | 0.600 | 0.582 | 0.554 | 0.526 |
| **GPT-4.1 (OpenAI)** | 0.548 | 0.700 | 0.600 | 0.676 | 0.524 |
| **DeepSeek V4 Pro** | 0.448 | 0.581 | 0.667 | 0.600 | 0.343 |

---

## Citation

If you use this benchmark or findings in your research, please cite our SBBD 2026 short paper:

```bibtex
@inproceedings{spatial_llm_sbbd2026,
  title={Geographic Inference with Large Language Models: Identification of POIs and Urban Features from Addresses and Coordinates},
  booktitle={Proceedings of the Brazilian Symposium on Databases (SBBD 2026)},
  year={2026}
}
```
