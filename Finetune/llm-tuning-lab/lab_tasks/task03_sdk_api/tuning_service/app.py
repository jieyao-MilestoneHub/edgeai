"""FastAPI 訓練服務"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import uuid

app = FastAPI(title="Tuning Service API")

# 簡化的任務儲存
jobs: Dict[str, Dict] = {}

class TuningRequest(BaseModel):
    model: str
    training_file: str
    hyperparameters: Dict = {}

@app.post("/v1/tunings.create")
async def create_tuning(request: TuningRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id,
        "model": request.model,
        "status": "pending",
        "hyperparameters": request.hyperparameters
    }
    return {"job_id": job_id, "status": "created"}

@app.get("/v1/tunings.get/{job_id}")
async def get_tuning(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]
