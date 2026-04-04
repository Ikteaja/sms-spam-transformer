###############################################################################
# SMS Spam Transformer — Makefile
# Usage: make <target>
###############################################################################

IMAGE   := ghcr.io/ikteaja/sms-spam-transformer
TAG     := $(shell git rev-parse --short HEAD 2>/dev/null || echo latest)
PORT    := 8000
PYTHON  := python

.DEFAULT_GOAL := help

.PHONY: help install data quality train evaluate serve \
        docker-build docker-run docker-push docker-scan \
        lint test sbom clean

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  SMS Spam Transformer"
	@echo "  ─────────────────────────────────────────────"
	@echo "  ML pipeline:"
	@echo "    make install     Install Python dependencies"
	@echo "    make data        Download + quality-check the dataset"
	@echo "    make train       Tokenise + fine-tune DistilBERT (all phases)"
	@echo "    make evaluate    Evaluate best model on validation set"
	@echo "    make test-final  One-shot final test evaluation (run once)"
	@echo "    make serve       Start API + Gradio UI  →  http://localhost:$(PORT)"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build  Build image  $(IMAGE):$(TAG)"
	@echo "    make docker-run    Run container on port $(PORT)"
	@echo "    make docker-push   Push to ghcr.io"
	@echo "    make docker-scan   Trivy vulnerability scan of image"
	@echo "    make sbom          Generate SBOM with Syft"
	@echo ""
	@echo "  Quality:"
	@echo "    make lint          flake8 + black check"
	@echo "    make test          Run pytest unit tests"
	@echo "    make clean         Remove generated data/model artefacts"
	@echo ""

# ── ML pipeline ───────────────────────────────────────────────────────────────
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install gradio>=4.0.0

data:
	$(PYTHON) scripts/01_download_data.py
	$(PYTHON) -c "\
import pandas as pd; \
df = pd.read_csv('data/raw/spam.csv', encoding='latin-1'); \
print('Rows:', len(df), '| Nulls:', df.isnull().sum().sum())"

quality:
	@echo "Running data quality checks..."
	$(PYTHON) -c "\
import pandas as pd, re; \
df = pd.read_csv('data/raw/spam.csv', encoding='latin-1'); \
df = df[['v1','v2']].rename(columns={'v1':'label','v2':'text'}) if 'v1' in df.columns else df[['label','text']]; \
assert df.isnull().sum().sum() == 0, 'FAIL: null values found'; \
assert set(df.label.str.strip().str.lower().unique()) == {'ham','spam'}, 'FAIL: invalid labels'; \
print('Data quality: OK')"

train: data quality
	$(PYTHON) scripts/03_tokenise.py
	$(PYTHON) scripts/04_train.py

evaluate:
	$(PYTHON) scripts/05_evaluate.py

freeze:
	$(PYTHON) scripts/06_freeze_tune.py

test-final:
	$(PYTHON) scripts/07_test_final.py

serve:
	uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build \
	  --label "org.opencontainers.image.revision=$(TAG)" \
	  -t $(IMAGE):$(TAG) \
	  -t $(IMAGE):latest \
	  .

docker-run:
	docker run --rm -it \
	  -p $(PORT):8000 \
	  -v "$(PWD)/models:/app/models:ro" \
	  -e MODEL_DIR=/app/models/frozen \
	  $(IMAGE):latest

docker-push:
	docker push $(IMAGE):$(TAG)
	docker push $(IMAGE):latest

docker-scan:
	@which trivy > /dev/null 2>&1 || (echo "Install trivy: https://aquasecurity.github.io/trivy" && exit 1)
	trivy image --exit-code 1 --severity HIGH,CRITICAL $(IMAGE):$(TAG)

sbom:
	@which syft > /dev/null 2>&1 || (echo "Install syft: https://github.com/anchore/syft" && exit 1)
	syft $(IMAGE):$(TAG) -o spdx-json > sbom.spdx.json
	@echo "SBOM saved to sbom.spdx.json"

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	flake8 app/ scripts/ tests/ --max-line-length=100 --ignore=E501,W503
	black --check app/ scripts/ tests/ --line-length=100

format:
	black app/ scripts/ tests/ --line-length=100

test:
	pytest tests/ -v --tb=short

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf data/processed models/checkpoints models/frozen_ckpt models/eval_tmp models/test_tmp
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned intermediate artefacts. data/raw and trained models preserved."
