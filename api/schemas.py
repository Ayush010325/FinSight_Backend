from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from datetime import date


class PredictionResponse(BaseModel):
    symbol: str
    direction: str
    signal: Optional[str] = None
    conviction: Optional[str] = None
    probability: float
    confidence: float
    prediction_date: date
    model_version: str


class BatchPredictionResponse(BaseModel):
    status: str
    prediction_date: date
    stocks_processed: int
    success_count: int
    failure_count: int
    execution_time_seconds: float
    results: List[dict]


class StockResponse(BaseModel):
    symbol: str
    company_name: str
    sector: Optional[str] = None
    is_active: bool


class TuningRequest(BaseModel):
    n_trials: int = 50
    timeout_seconds: Optional[int] = None
    model_version: str = "classification_v2"


class TuningResponse(BaseModel):
    status: str
    best_trial: int
    best_accuracy: float
    final_retrain_accuracy: float
    best_config: Dict[str, Any]
    elapsed_seconds: float
    final_retrain_per_stock: Dict[str, float]
