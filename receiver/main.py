"""
FastAPI Backend for IoT Telemetry Data Ingestion
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import sqlite3
import os

# Initialize FastAPI app
app = FastAPI(
    title="IoT Telemetry Receiver",
    description="Endpoint for ingesting live IoT sensor data",
    version="1.0.0"
)

# Database setup
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "telemetry.db")

# Create SQLite engine and session factory
engine = create_engine(f"sqlite:///{DATABASE_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()


class TelemetryRecord(Base):
    """SQLAlchemy model for telemetry records"""
    __tablename__ = "telemetry_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    node_id = Column(Text, nullable=False)
    soil_moisture_percent = Column(Float, nullable=False)
    nitrogen_mg = Column(Float, nullable=False)
    phosphorus_mg = Column(Float, nullable=False)
    potassium_mg = Column(Float, nullable=False)
    ambient_temp_c = Column(Float, nullable=False)
    created_at = Column(Text, nullable=False)


# Pydantic models for request validation
class TelemetryPayload(BaseModel):
    """Pydantic model for incoming telemetry data"""
    node_id: str = Field(..., description="Unique identifier for the IoT sensor node")
    soil_moisture_percent: float = Field(..., ge=0.0, le=100.0, description="Soil moisture percentage (0-100%)")
    nitrogen_mg: float = Field(..., ge=0.0, description="Nitrogen concentration in mg")
    phosphorus_mg: float = Field(..., ge=0.0, description="Phosphorus concentration in mg")
    potassium_mg: float = Field(..., ge=0.0, description="Potassium concentration in mg")
    ambient_temp_c: float = Field(..., description="Ambient temperature in Celsius")


# Create database tables on startup
def init_db():
    """Initialize the database and create tables"""
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully")


# Initialize database at startup
init_db()


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "IoT Telemetry Receiver"}


@app.post(
    "/api/telemetry",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    tags=["Telemetry"]
)
async def receive_telemetry(payload: TelemetryPayload):
    """
    Ingest live IoT sensor telemetry data

    Accepts soil moisture, nutrient levels, and temperature readings from IoT nodes.
    Stores records in SQLite database for later processing.
    """
    try:
        # Create a new database session
        db = SessionLocal()

        try:
            # Convert Pydantic model to SQLAlchemy model
            record = TelemetryRecord(
                node_id=payload.node_id,
                soil_moisture_percent=payload.soil_moisture_percent,
                nitrogen_mg=payload.nitrogen_mg,
                phosphorus_mg=payload.phosphorus_mg,
                potassium_mg=payload.potassium_mg,
                ambient_temp_c=payload.ambient_temp_c,
                created_at=str(payload.model_dump())
            )

            # Add record to database
            db.add(record)
            db.commit()
            db.refresh(record)

            # Return success response with stored data
            return {
                "status": "received",
                "record_id": record.id,
                "node_id": record.node_id,
                "data": {
                    "soil_moisture_percent": record.soil_moisture_percent,
                    "nitrogen_mg": record.nitrogen_mg,
                    "phosphorus_mg": record.phosphorus_mg,
                    "potassium_mg": record.potassium_mg,
                    "ambient_temp_c": record.ambient_temp_c
                }
            }

        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid payload: {str(e)}"
        )


@app.get("/api/telemetry", tags=["Telemetry"])
async def get_telemetry_records(limit: int = 100, offset: int = 0):
    """
    Retrieve stored telemetry records from the database

    Args:
        limit: Maximum number of records to return
        offset: Number of records to skip for pagination
    """
    try:
        db = SessionLocal()

        try:
            records = db.query(TelemetryRecord).offset(offset).limit(limit).all()

            return {
                "count": len(records),
                "records": [
                    {
                        "id": r.id,
                        "node_id": r.node_id,
                        "soil_moisture_percent": r.soil_moisture_percent,
                        "nitrogen_mg": r.nitrogen_mg,
                        "phosphorus_mg": r.phosphorus_mg,
                        "potassium_mg": r.potassium_mg,
                        "ambient_temp_c": r.ambient_temp_c,
                        "created_at": r.created_at
                    }
                    for r in records
                ]
            }

        finally:
            db.close()

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query error: {str(e)}"
        )


@app.get("/api/telemetry/stats", tags=["Telemetry"])
async def get_telemetry_stats():
    """Get statistics about stored telemetry data"""
    try:
        db = SessionLocal()

        try:
            total_records = db.query(TelemetryRecord).count()

            if total_records == 0:
                return {
                    "total_records": 0,
                    "message": "No records in database"
                }

            stats = {
                "total_records": total_records,
                "unique_nodes": db.query(TelemetryRecord).distinct().count(TelemetryRecord.node_id),
                "avg_soil_moisture": db.query(
                    db.func.avg(TelemetryRecord.soil_moisture_percent)
                ).scalar() or 0,
                "avg_nitrogen_mg": db.query(
                    db.func.avg(TelemetryRecord.nitrogen_mg)
                ).scalar() or 0,
                "avg_phosphorus_mg": db.query(
                    db.func.avg(TelemetryRecord.phosphorus_mg)
                ).scalar() or 0,
                "avg_potassium_mg": db.query(
                    db.func.avg(TelemetryRecord.potassium_mg)
                ).scalar() or 0,
                "avg_ambient_temp_c": db.query(
                    db.func.avg(TelemetryRecord.ambient_temp_c)
                ).scalar() or 0,
            }

            return stats

        finally:
            db.close()

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Statistics query error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
