# Spatial LLM Benchmark

Use the separate runner files for simple execution.

## Simple runners

### OpenAI only

Set `LLM_API_KEY` in `.env`, then run:

```powershell
python run_openai.py
```

Optional:

```powershell
python run_openai.py --model gpt-4o-mini --input addresses_custom.csv --output results_openai.json
```

To avoid rate limits, process slowly or test just a few rows:

```powershell
python run_openai.py --delay 10 --limit 1
```

### Ollama only

Start Ollama locally, then run:

```powershell
python run_ollama.py
```

Optional:

```powershell
python run_ollama.py --model llama3:8b --input addresses_custom.csv --output results_ollama.json
```

Both files generate only model predictions. They do not call OSM and do not run the judge.

### Gemini grouped by country

Set `GEMINI_API_KEY` in `.env`, then run:

```powershell
python run_gemini_addresses_by_country.py
```

Optional:

```powershell
python run_gemini_addresses_by_country.py --model gemini-3-flash-preview --input csv/addresses.csv --delay 2
```

This runner uses `prompts/address_user_prompt.txt`, temperature `0`, and saves predictions grouped by country.

## Validate OpenAI Results With Ollama + OSM

After generating `results_openai.json`, start Ollama locally and run:

```powershell
python validate_openai_with_ollama_osm.py --input results_openai.json --output validation_openai_ollama_osm.json
```

To test only one row first:

```powershell
python validate_openai_with_ollama_osm.py --input results_openai.json --limit 1
```

To validate distance as route distance through OpenRouteService, set `ORS_API_KEY` in `.env` and run:

```powershell
python validate_openai_with_ollama_osm.py --input results_openai.json --distance-mode route --ors-profile foot-walking
```

Use `--ors-profile driving-car` if you want car travel distance instead.

When `--distance-mode route` is used without `ORS_API_KEY`, validation falls back to straight-line distance and records `distance_mode` as `straight_fallback` in the output JSON.

You can run the unit tests for the ORS integration with:

```powershell
python -m unittest discover -s tests
```

## Advanced legacy runner

`main.py` keeps the old benchmark flow.

## Supported providers

- `ollama`
- `openai`
- `deepseek`
- `anthropic`
- `gemini`
- `openai_compatible`

## Main environment variables

- `LLM_PROVIDER`
- `LLM_MODEL`
- `LLM_API_KEY`
- `LLM_BASE_URL` (optional)
- `JUDGE_PROVIDER` (optional)
- `JUDGE_MODEL` (optional)
- `JUDGE_API_KEY` (optional)
- `JUDGE_BASE_URL` (optional)

If judge variables are not provided, the benchmark reuses the same provider/model from the main generation step.

## Examples

### OpenAI

```powershell
$env:LLM_PROVIDER="openai"
$env:LLM_MODEL="gpt-4o-mini"
$env:LLM_API_KEY="YOUR_OPENAI_KEY"
python main.py
```

### Gemini

```powershell
$env:LLM_PROVIDER="gemini"
$env:LLM_MODEL="gemini-1.5-flash"
$env:LLM_API_KEY="YOUR_GEMINI_KEY"
python main.py
```

### Claude Sonnet

```powershell
$env:LLM_PROVIDER="anthropic"
$env:LLM_MODEL="claude-3-5-sonnet-latest"
$env:LLM_API_KEY="YOUR_ANTHROPIC_KEY"
python main.py
```

### DeepSeek

```powershell
$env:LLM_PROVIDER="deepseek"
$env:LLM_MODEL="deepseek-chat"
$env:LLM_API_KEY="YOUR_DEEPSEEK_KEY"
python main.py
```

### Ollama

```powershell
$env:LLM_PROVIDER="ollama"
$env:LLM_MODEL="llama3:8b"
python main.py
```

## Running

```powershell
python main.py --input addresses.csv --output final_results.json
```

The default `addresses.csv` already contains the city/scenario benchmark set.
