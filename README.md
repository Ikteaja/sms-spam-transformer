# SMS Spam Classifier — Transfer Learning with DistilBERT

Fine-tune DistilBERT on the UCI SMS Spam Collection dataset.
Full lifecycle: data → tokenise → train → evaluate → freeze → deploy → track.

## Project layout

```
sms-spam-transformer/
├── data/                    # raw + processed datasets (git-ignored)
├── models/                  # saved checkpoints (git-ignored)
├── mlruns/                  # MLflow experiment logs (git-ignored)
├── scripts/
│   ├── 01_download_data.py  # Phase 1 – fetch dataset from Kaggle
│   ├── 02_explore_data.py   # Phase 2 – EDA, class balance, text stats
│   ├── 03_tokenise.py       # Phase 3 – DistilBERT tokenisation + 70/15/15 split
│   ├── 04_train.py          # Phase 4 – fine-tune DistilBERT (3 epochs)
│   ├── 05_evaluate.py       # Phase 5 – metrics, confusion matrix
│   ├── 06_freeze_tune.py    # Phase 6 – freeze lower layers, re-tune head
│   └── 07_test_final.py     # Phase 7 – one-shot final test set evaluation
├── app/
│   └── main.py              # Phase 8 – FastAPI inference server
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Put your Kaggle credentials in ~/.kaggle/kaggle.json  OR  drop
#    spam.csv into data/raw/ manually.
python scripts/01_download_data.py

# 2-3. Explore and tokenise
python scripts/02_explore_data.py
python scripts/03_tokenise.py

# 4. Train  (≈10 min on CPU)
python scripts/04_train.py

# 5-7. Evaluate → freeze-tune → final test
python scripts/05_evaluate.py
python scripts/06_freeze_tune.py
python scripts/07_test_final.py

# 8. Serve
uvicorn app.main:app --reload

# MLflow UI
mlflow ui          # open http://localhost:5000
```

## Why these choices?

| Decision | Reason |
|---|---|
| DistilBERT | 66 MB, trains in ~10 min on laptop CPU, same accuracy as BERT for classification |
| SMS Spam dataset | 5 574 rows, clean binary labels, free |
| Transfer learning | Only 3.5 % of parameters trained when layers are frozen |
| MLflow | Free, local, zero-config experiment tracking |
| FastAPI | Async, typed, auto-docs at `/docs` |
