# Smart Scan Strategy — Demo Dashboard

## Run locally

```bash
cd backend
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000

The dashboard serves the existing frozen ML checkpoints and validation RF environments. It does not retrain or modify the ML notebooks.
