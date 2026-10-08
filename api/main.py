from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
from datetime import date, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SessionLocal
from database.models_db import Stock, PredictionRecord
from services.prediction_service import predict_all_stocks
from services.tuning_service import run_hyperparameter_tuning
from api.schemas import PredictionResponse, BatchPredictionResponse, StockResponse, TuningRequest, TuningResponse

app = FastAPI(title="FinSight API", description="API for FinSight Stock Prediction Backend", version="1.0.0")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_target_trading_date(current_date: date) -> date:
    if current_date.weekday() == 4:
        return current_date + timedelta(days=3)
    elif current_date.weekday() == 5:
        return current_date + timedelta(days=2)
    elif current_date.weekday() == 6:
        return current_date + timedelta(days=1)
    return current_date + timedelta(days=1)


@app.get("/")
def health_check():
    return {"status": "healthy", "service": "FinSight API"}


@app.get("/stocks", response_model=List[StockResponse])
def get_active_stocks(db: Session = Depends(get_db)):
    return db.query(Stock).filter(Stock.is_active == True).all()


@app.get("/predictions", response_model=List[PredictionResponse])
def get_all_latest_predictions(db: Session = Depends(get_db)):
    stocks = db.query(Stock).filter(Stock.is_active == True).all()
    results = []
    for stock in stocks:
        pred = db.query(PredictionRecord).filter(
            PredictionRecord.symbol == stock.symbol
        ).order_by(PredictionRecord.prediction_date.desc()).first()

        if pred and pred.predictions:
            results.append(PredictionResponse(
                symbol=pred.symbol,
                direction=pred.predictions.get("direction"),
                probability=pred.predictions.get("probability"),
                confidence=pred.predictions.get("confidence"),
                prediction_date=pred.prediction_date,
                model_version=pred.model_version
            ))
    return results


@app.get("/predictions/{symbol}", response_model=PredictionResponse)
def get_prediction_for_stock(symbol: str, db: Session = Depends(get_db)):
    pred = db.query(PredictionRecord).filter(
        PredictionRecord.symbol == symbol
    ).order_by(PredictionRecord.prediction_date.desc()).first()

    if not pred or not pred.predictions:
        raise HTTPException(status_code=404, detail=f"No predictions found for {symbol}")

    return PredictionResponse(
        symbol=pred.symbol,
        direction=pred.predictions.get("direction"),
        probability=pred.predictions.get("probability"),
        confidence=pred.predictions.get("confidence"),
        prediction_date=pred.prediction_date,
        model_version=pred.model_version
    )


@app.post("/predictions/run", response_model=BatchPredictionResponse)
def run_prediction_pipeline(db: Session = Depends(get_db)):
    try:
        target_date = get_target_trading_date(date.today())
        result = predict_all_stocks(db=db, prediction_date=target_date)
        if result.get("status") == "failed":
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result

@app.post("/training/tune", response_model=TuningResponse)
def trigger_hyperparameter_tuning(
    request: TuningRequest,
    db: Session = Depends(get_db),
):
    try:
        report = run_hyperparameter_tuning(
            db=db,
            n_trials=request.n_trials,
            timeout_seconds=request.timeout_seconds,
            best_model_version=request.model_version,
        )
        return {
            "status": "completed",
            "best_trial": report["best_trial"],
            "best_accuracy": report["best_accuracy"],
            "final_retrain_accuracy": report["final_retrain_accuracy"],
            "best_config": report["best_config"],
            "elapsed_seconds": report["elapsed_seconds"],
            "final_retrain_per_stock": report["final_retrain_per_stock"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

