# Sparse Code Prediction

Predict geospatial properties from location using either location embeddings (GeoCLIP, SatCLIP) or sparse concept-based embeddings (SPLICE, soon to add SAE support). A ridge regression is fit on top of the embeddings and evaluated on held-out test data. When using sparse embeddings, we report the top contributing concepts.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For SatCLIP, clone the repo into `external/` (should already be there right now):

> **Note:** I had to adjust some imports in the SatCLIP repo to get it working. Pushing the clone directly is easier for now.

### 2. Download datasets

**EuroSAT:**
```bash
cd datasets
python EuroSAT.py --download --root /data/eurosat
```

**MOSAIKS** (elevation, tree cover, nightlights, population density): I put these on google drive ([link](https://drive.google.com/file/d/1qLLSo8bbKCCGo9voWCNsorCaJEoV7yDz/view?usp=sharing)) for now (also available on code ocean or TorchSpatial). Note however that this does not have images, but I put because it's easy to work with.

**Others**
I'm currently working on SustainBench + GeoYFCC.

### 3. Configure paths

Update `paths.yaml` with the paths to your data and SPLICE model checkpoints:

```yaml
datasets:
  eurosat: /data/eurosat
  mosaiks:
    elevation: /data/mosaiks/elevation
    tree_cover: /data/mosaiks/treecover
    nightlights: /data/mosaiks/nightlights
    population_density: /data/mosaiks/population
```

## Usage

### Run ridge regression

**With location embeddings (GeoCLIP or SatCLIP for now):**
```bash
python main.py --dataset eurosat --embeddings location --encoder geoclip
python main.py --dataset elevation --embeddings location --encoder satclip
```

**With sparse SpLiCE embeddings:**
```bash
python main.py --dataset eurosat --embeddings splice --splice_model geoclip_geoyfcc
```

**With sparse SAE embeddings:**
```bash
python main.py --dataset eurosat --embeddings sae --sae_model_path /path/to/sae.pt
```

This prints R² on the test set, and for sparse embeddings also prints the top contributing concepts.

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | required | `eurosat`, `elevation`, `tree_cover`, `nightlights`, `population_density` |
| `--embeddings` | `location` | `location`, `splice`, or `sae` |
| `--encoder` | — | `geoclip` or `satclip` (required for `location`) |
| `--splice_model` | — | Key from `paths.yaml` (required for `splice`) |
| `--sae_model_path` | — | Path to SAE `.pt` checkpoint (required for `sae`) |
| `--topk` | `10` | Number of top concepts to report |
| `--alpha` | `1.0` | Ridge regularization strength |
| `--cv` | off | Use cross-validated alpha selection (`RidgeCV`) |
| `--embeddings_dir` | dataset dir | Where to cache/load precomputed embeddings |
| `--output_dir` | `outputs/` | Where to save the fitted ridge model |

### Pre-generate embeddings separately

You can generate and cache embeddings before running the regression:

```bash
# Location embeddings
python earth_embeddings.py --encoder geoclip --dataset eurosat

# Sparse SpLiCE embeddings
python splice_embeddings.py --splice_model geoclip_geoyfcc --dataset eurosat
```

Embeddings are saved to the dataset directory (or `--output_dir`) and loaded automatically on subsequent runs.

## LocationEncoder note

`models/location_encoder.py` provides a unified wrapper around GeoCLIP and SatCLIP. Both expect `(lat, lon)` as input — the wrapper internally reorders to `(lon, lat)` for SatCLIP. IMPORTANT: Do not do this reordering yourself IF you use the wrapper.
