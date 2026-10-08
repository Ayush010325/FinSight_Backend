from sqlalchemy.orm import Session
from datetime import date, datetime
from typing import Dict, Any, List, Optional
from database.models_db import ModelRecord, TrainingJobRecord, PredictionRecord


def create_model_record(
    db: Session,
    symbol: str,
    model_version: str,
    model_path: str,
    scaler_path: str,
    accuracy: float,
    trained_until: date,
    is_active: bool = True
) -> ModelRecord:
    existing = db.query(ModelRecord).filter(
        ModelRecord.model_version == model_version
    ).first()

    if existing:
        existing.symbol = symbol
        existing.model_path = model_path
        existing.scaler_path = scaler_path
        existing.accuracy = accuracy
        existing.trained_until = trained_until
        existing.is_active = is_active
        db.commit()
        db.refresh(existing)
        return existing

    db_model = ModelRecord(
        symbol=symbol,
        model_version=model_version,
        model_path=model_path,
        scaler_path=scaler_path,
        accuracy=accuracy,
        trained_until=trained_until,
        is_active=is_active
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model


def get_active_model(db: Session, symbol: str) -> Optional[ModelRecord]:
    return db.query(ModelRecord).filter(
        ModelRecord.symbol == symbol,
        ModelRecord.is_active == True
    ).order_by(ModelRecord.model_version.desc()).first()

def deactivate_old_models(db: Session, symbol: str, active_version: str) -> None:
    db.query(ModelRecord).filter(
        ModelRecord.symbol == symbol,
        ModelRecord.model_version != active_version
    ).update({ModelRecord.is_active: False}, synchronize_session=False)
    db.commit()

def save_prediction(
    db: Session,
    symbol: str,
    model_version: str,
    prediction_date: date,
    predictions_dict: Dict[str, str]
) -> PredictionRecord:
    db_prediction = PredictionRecord(
        symbol=symbol,
        model_version=model_version,
        prediction_date=prediction_date,
        predictions=predictions_dict
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

def get_prediction_history(db: Session, symbol: str, limit: int = 10) -> List[PredictionRecord]:
    return db.query(PredictionRecord).filter(
        PredictionRecord.symbol == symbol
    ).order_by(PredictionRecord.prediction_date.desc()).limit(limit).all()

def create_training_run(
    db: Session,
    symbol: str,
    model_version: str,
    epochs: int,
    train_samples: int,
    validation_accuracy: float,
    loss: float,
    started_at: datetime,
    completed_at: datetime
) -> TrainingJobRecord:
    db_run = TrainingJobRecord(
        symbol=symbol,
        model_version=model_version,
        status="completed",
        epochs=epochs,
        train_samples=train_samples,
        validation_accuracy=validation_accuracy,
        loss=loss,
        started_at=started_at,
        completed_at=completed_at
    )
    db.add(db_run)
    db.commit()
    db.refresh(db_run)
    return db_run

def start_training_run(
    db: Session,
    symbol: str,
    model_version: str,
    started_at: datetime = None
) -> TrainingJobRecord:
    db_run = TrainingJobRecord(
        symbol=symbol,
        model_version=model_version,
        status="running",
        started_at=started_at or datetime.utcnow()
    )
    db.add(db_run)
    db.commit()
    db.refresh(db_run)
    return db_run

def complete_training_run(
    db: Session,
    run_id: int,
    epochs: int,
    train_samples: int,
    validation_accuracy: float,
    loss: float,
    completed_at: datetime = None
) -> Optional[TrainingJobRecord]:
    db_run = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == run_id).first()
    if db_run:
        db_run.status = "completed"
        db_run.epochs = epochs
        db_run.train_samples = train_samples
        db_run.validation_accuracy = validation_accuracy
        db_run.loss = loss
        db_run.completed_at = completed_at or datetime.utcnow()
        db.commit()
        db.refresh(db_run)
    return db_run

def fail_training_run(
    db: Session,
    run_id: int,
    completed_at: datetime = None
) -> Optional[TrainingJobRecord]:
    db_run = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == run_id).first()
    if db_run:
        db_run.status = "failed"
        db_run.completed_at = completed_at or datetime.utcnow()
        db.commit()
        db.refresh(db_run)
    return db_run

def register_successful_training(
    db: Session,
    symbol: str,
    model_version: str,
    model_path: str,
    scaler_path: str,
    accuracy: float,
    trained_until: date,
    run_id: int = None,
    epochs: int = None,
    train_samples: int = None,
    loss: float = None,
    started_at: datetime = None,
    completed_at: datetime = None
) -> ModelRecord:
    deactivate_old_models(db=db, symbol=symbol, active_version=model_version)
    model_record = create_model_record(
        db=db,
        symbol=symbol,
        model_version=model_version,
        model_path=model_path,
        scaler_path=scaler_path,
        accuracy=accuracy,
        trained_until=trained_until,
        is_active=True
    )
    if run_id is not None:
        complete_training_run(
            db=db,
            run_id=run_id,
            epochs=epochs or 0,
            train_samples=train_samples or 0,
            validation_accuracy=accuracy,
            loss=loss or 0.0,
            completed_at=completed_at
        )
    elif epochs is not None or train_samples is not None or loss is not None:
        create_training_run(
            db=db,
            symbol=symbol,
            model_version=model_version,
            epochs=epochs,
            train_samples=train_samples,
            validation_accuracy=accuracy,
            loss=loss,
            started_at=started_at or datetime.utcnow(),
            completed_at=completed_at or datetime.utcnow()
        )
    return model_record

def get_model_history(db: Session, symbol: str) -> List[ModelRecord]:
    return db.query(ModelRecord).filter(
        ModelRecord.symbol == symbol
    ).order_by(ModelRecord.created_at.desc()).all()


def get_training_runs(db: Session, symbol: str) -> List[TrainingJobRecord]:
    return db.query(TrainingJobRecord).filter(
        TrainingJobRecord.symbol == symbol
    ).order_by(TrainingJobRecord.started_at.desc()).all()


def get_latest_prediction(
    db: Session,
    symbol: str,
    model_version_prefix: str = None,
) -> Optional[PredictionRecord]:
    query = db.query(PredictionRecord).filter(PredictionRecord.symbol == symbol)
    if model_version_prefix:
        query = query.filter(PredictionRecord.model_version.like(f"{model_version_prefix}%"))
    return query.order_by(PredictionRecord.prediction_date.desc()).first()


def get_prediction_history(
    db: Session,
    symbol: str,
    limit: int = 30,
    model_version_prefix: str = None,
) -> List[PredictionRecord]:
    query = db.query(PredictionRecord).filter(PredictionRecord.symbol == symbol)
    if model_version_prefix:
        query = query.filter(PredictionRecord.model_version.like(f"{model_version_prefix}%"))
    return query.order_by(PredictionRecord.prediction_date.desc()).limit(limit).all()


def get_model_used_for_prediction(
    db: Session,
    symbol: str,
    prediction_date: date,
) -> Optional[ModelRecord]:
    pred = (
        db.query(PredictionRecord)
        .filter(
            PredictionRecord.symbol == symbol,
            PredictionRecord.prediction_date == prediction_date,
        )
        .order_by(PredictionRecord.created_at.desc())
        .first()
    )
    if pred is None:
        return None
    return db.query(ModelRecord).filter(
        ModelRecord.model_version == pred.model_version
    ).first()
