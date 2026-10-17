
from dotenv import load_dotenv
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from datetime import datetime
import os
import pickle
import logging
import math
import pandas as pd
import mysql.connector


# ==========================================
# FLASK APPLICATION CONFIGURATION
# ==========================================

app = Flask(__name__)
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# LOAD TRAINED ML MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Model" / "OEE_Tomorrow_Model.pkl"

with open(MODEL_PATH, "rb") as file:
    saved_data = pickle.load(file)

model = saved_data["model"]
features = list(saved_data["features"])

logger.info("OEE Model loaded successfully!")
logger.info("Features: %s", features)


# ==========================================
# AZURE MYSQL DATABASE CONNECTION
# ==========================================

def get_db_connection():

    config = {
        "host": os.environ["DB_HOST"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "database": os.environ["DB_NAME"],
        "port": int(os.getenv("DB_PORT", "3306")),
        "ssl_disabled": False,
        "connection_timeout": 15
    }

    # Optional CA certificate configuration
    ssl_ca = os.getenv("DB_SSL_CA")

    if ssl_ca:
        config["ssl_ca"] = ssl_ca
        config["ssl_verify_cert"] = True
        config["ssl_verify_identity"] = True

    return mysql.connector.connect(**config)


# ==========================================
# INITIALIZE AZURE MYSQL TABLE
# ==========================================

def initialize_database():

    connection = None
    cursor = None

    try:
        logger.info("Initializing Azure MySQL database...")

        connection = get_db_connection()
        cursor = connection.cursor()

        create_table_query = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            machine_age INT,
            oee_lag_1 FLOAT,
            avail_hrs_lag_1 FLOAT,
            mc_running_hrs_lag_1 FLOAT,
            total_dt_lag_1 FLOAT,
            prod_qty_lag_1 INT,
            rej_qty_lag_1 INT,
            maintenance_flag_lag_1 INT,
            predicted_oee FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        cursor.execute(create_table_query)
        connection.commit()

        logger.info("Azure MySQL predictions table is ready!")

        return True

    except mysql.connector.Error:
        logger.exception("Database initialization failed")
        return False

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None and connection.is_connected():
            connection.close()


# ==========================================
# INITIALIZE DATABASE ON STARTUP
# ==========================================

# CREATE TABLE IF NOT EXISTS preserves old records.
# This operation is safe to repeat during deployments.

database_ready = initialize_database()

if not database_ready:
    logger.warning(
        "Database initialization failed. "
        "Prediction requests may fail until MySQL is available."
    )


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return render_template("index.html")


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "running",
        "service": "OEE Tomorrow Prediction API",
        "model_loaded": True
    }), 200


# ==========================================
# REQUIRED INPUT FEATURES
# ==========================================

required_fields = [
    "Machine_Age",
    "OEE_Lag_1",
    "Avail_Hrs_Lag_1",
    "MC_Running_Hrs_Lag_1",
    "Total_DT_Lag_1",
    "Prod_Qty_Lag_1",
    "Rej_Qty_Lag_1",
    "Maintenance_Flag_Lag_1"
]

integer_fields = [
    "Machine_Age",
    "Prod_Qty_Lag_1",
    "Rej_Qty_Lag_1",
    "Maintenance_Flag_Lag_1"
]


# ==========================================
# PREDICT OEE
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    connection = None
    cursor = None

    try:

        # ----------------------------------
        # GET JSON INPUT
        # ----------------------------------

        data = request.get_json(silent=True)

        if not isinstance(data, dict) or not data:
            return jsonify({
                "error": "Valid JSON object is required"
            }), 400

        # ----------------------------------
        # CHECK MISSING FIELDS
        # ----------------------------------

        missing_fields = [
            field for field in required_fields
            if field not in data
        ]

        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        # ----------------------------------
        # CHECK NUMERIC DATA TYPES
        # ----------------------------------

        for field in required_fields:

            value = data[field]

            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                return jsonify({
                    "error": "Invalid data type",
                    "field": field,
                    "message": f"{field} must be numeric"
                }), 400

            if not math.isfinite(value):
                return jsonify({
                    "error": "Invalid numeric value",
                    "field": field
                }), 400

        # ----------------------------------
        # CHECK INTEGER FIELDS
        # ----------------------------------

        for field in integer_fields:

            if not isinstance(data[field], int):
                return jsonify({
                    "error": "Invalid data type",
                    "field": field,
                    "message": f"{field} must be an integer"
                }), 400

        # ----------------------------------
        # CHECK NON-NEGATIVE VALUES
        # ----------------------------------

        non_negative_fields = [
            "Machine_Age",
            "Avail_Hrs_Lag_1",
            "MC_Running_Hrs_Lag_1",
            "Total_DT_Lag_1",
            "Prod_Qty_Lag_1",
            "Rej_Qty_Lag_1"
        ]

        for field in non_negative_fields:

            if data[field] < 0:
                return jsonify({
                    "error": "Invalid value",
                    "field": field,
                    "message": f"{field} cannot be negative"
                }), 400

        # ----------------------------------
        # CHECK OEE RANGE
        # ----------------------------------

        if not 0 <= data["OEE_Lag_1"] <= 100:
            return jsonify({
                "error": "Invalid value",
                "field": "OEE_Lag_1",
                "message": "OEE must be between 0 and 100"
            }), 400

        # ----------------------------------
        # CHECK MAINTENANCE FLAG
        # ----------------------------------

        if data["Maintenance_Flag_Lag_1"] not in [0, 1]:
            return jsonify({
                "error": "Invalid maintenance flag",
                "message": "Maintenance flag must be 0 or 1"
            }), 400

        # ----------------------------------
        # PREPARE DATAFRAME
        # ----------------------------------

        input_data = pd.DataFrame(
            [{field: data[field] for field in required_fields}]
        )

        # Arrange features in model training order
        input_data = input_data[features]

        # ----------------------------------
        # MAKE MODEL PREDICTION
        # ----------------------------------

        prediction = model.predict(input_data)

        predicted_oee = round(
            float(prediction[0]), 2
        )

        if not math.isfinite(predicted_oee):
            raise ValueError("Model returned non-finite prediction")

        logger.info("Predicted OEE: %s", predicted_oee)

        # ----------------------------------
        # CONNECT TO AZURE MYSQL
        # ----------------------------------

        connection = get_db_connection()
        cursor = connection.cursor()

        # ----------------------------------
        # SAVE PREDICTION TO AZURE MYSQL
        # ----------------------------------

        insert_query = """
        INSERT INTO predictions (
            machine_age,
            oee_lag_1,
            avail_hrs_lag_1,
            mc_running_hrs_lag_1,
            total_dt_lag_1,
            prod_qty_lag_1,
            rej_qty_lag_1,
            maintenance_flag_lag_1,
            predicted_oee
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s
        )
        """

        values = (
            data["Machine_Age"],
            data["OEE_Lag_1"],
            data["Avail_Hrs_Lag_1"],
            data["MC_Running_Hrs_Lag_1"],
            data["Total_DT_Lag_1"],
            data["Prod_Qty_Lag_1"],
            data["Rej_Qty_Lag_1"],
            data["Maintenance_Flag_Lag_1"],
            predicted_oee
        )

        cursor.execute(insert_query, values)
        connection.commit()

        prediction_id = cursor.lastrowid

        logger.info(
            "Prediction saved to Azure MySQL, ID: %s",
            prediction_id
        )

        # ----------------------------------
        # RETURN RESPONSE
        # ----------------------------------

        return jsonify({
            "message": "Prediction successful and saved to Azure MySQL",
            "prediction_id": prediction_id,
            "predicted_oee": predicted_oee
        }), 200

    except mysql.connector.Error:

        if connection is not None:
            connection.rollback()

        logger.exception("Azure MySQL operation failed")

        return jsonify({
            "error": "Database operation failed"
        }), 503

    except Exception:

        logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction processing failed"
        }), 500

    finally:

        if cursor is not None:
            cursor.close()

        if connection is not None and connection.is_connected():
            connection.close()


# ==========================================
# GET SAVED PREDICTIONS
# ==========================================

@app.route("/predictions", methods=["GET"])
def get_predictions():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT *
            FROM predictions
            ORDER BY created_at DESC, id DESC
            LIMIT 100
        """)

        records = cursor.fetchall()

        # Convert datetime to JSON-compatible format
        for record in records:

            if isinstance(record.get("created_at"), datetime):
                record["created_at"] = (
                    record["created_at"].isoformat()
                )

        return jsonify(records), 200

    except mysql.connector.Error:

        logger.exception("Unable to retrieve predictions")

        return jsonify({
            "error": "Unable to fetch predictions"
        }), 503

    finally:

        if cursor is not None:
            cursor.close()

        if connection is not None and connection.is_connected():
            connection.close()


# ==========================================
# RUN FLASK LOCALLY
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False
    )
