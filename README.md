# **"Geographic Inference with Large Language Models: Identification of POIs and Urban Features from Addresses and Coordinates"** (Brazilian Symposium on Databases - SBBD 2026)

This repository contains the dataset, source code, and evaluation results.

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

## Citation

If you use this benchmark or findings in your research, please cite our SBBD 2026 short paper:

```bibtex
@INPROCEEDINGS{249658,
    AUTHOR="Gabriel Silva and Salatiel Silva and Claudio Campelo",
    TITLE="Geographic Inference with Large Language Models: Identification of POIs and Urban Features from Addresses and Coordinates",
    BOOKTITLE="SBBD 2026 - Short Papers () ",
    ADDRESS="",
    DAYS="8-11",
    MONTH="sep",
    YEAR="2026",
    ABSTRACT="This paper evaluates the ability of Large Language Models (LLMs) to infer geographic features and identify points of interest (POIs) from addresses and coordinates. Nine cities of different urban sizes, distributed across three continents, were analyzed, with five addresses per city in scenarios such as urban center, residential area, green area, proximity to water and mixed area. The predictions were validated with external geospatial databases. The results indicate greater stability in feature inference than in POI identification, whose performance varies according to country and urban size; Gemini 3.1 Flash Lite leads in large and medium-sized cities, with a tie between Gemini 3 Flash Preview and DeepSeek V4 Flash in small cities.",
    KEYWORDS="Experimentos e análises; Aprendizado de máquina, IA, gerenciamento de dados e sistemas de dados",
    URL="http://XXXXX/249658.pdf"
}
```
