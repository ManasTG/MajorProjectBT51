
from sqlalchemy import inspect, text
from app.database import engine

new_columns = {
    "total_samples": "INTEGER",
    "sampling_rate": "INTEGER",
    "class_counts": "JSON",
    "class_percentages": "JSON",
    "abnormal_beats": "INTEGER",
    "average_confidence": "FLOAT",
    "beat_predictions": "JSON",
    "r_peak_indices": "JSON",
    "filtered_signal": "JSON"
}

inspector = inspect(engine)

if "ecg_analyses" not in inspector.get_table_names():
    raise RuntimeError("ecg_analyses table not found")

existing_columns = {
    column["name"]
    for column in inspector.get_columns("ecg_analyses")
}

with engine.begin() as connection:
    for column_name, column_type in new_columns.items():
        if column_name not in existing_columns:
            connection.execute(
                text(
                    f"ALTER TABLE ecg_analyses "
                    f"ADD COLUMN {column_name} {column_type}"
                )
            )
            print(f"Added: {column_name}")
        else:
            print(f"Already exists: {column_name}")

print("\nDatabase migration completed successfully!")
