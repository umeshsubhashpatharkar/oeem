        try:
            cursor.execute(CREATE_TABLE_SQL)
            connection.commit()
            _table_ready = True
            logger.info("Azure MySQL predictions table is ready")
        finally:
            cursor.close()


def initialize_database():
    """Optional explicit initialization; also happens automatically on first DB call."""
    connection = get_db_connection()
    try:
        ensure_predictions_table(connection)
        return True
    finally:
        connection.close()


def validate_input(data):
    if not isinstance(data, dict) or not data:
        return {"error": "A JSON object is required"}
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        return {"error": "Missing required fields", "missing_fields": missing}

    for field in REQUIRED_FIELDS:
        value = data[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return {"error": "Invalid data type/value", "field": field, "message": "A finite number is required"}
        if field in INTEGER_FIELDS and not isinstance(value, int):
            return {"error": "Invalid data type", "field": field, "message": "An integer is required"}
        if field in NONNEGATIVE_FIELDS and value < 0:
            return {"error": "Invalid value", "field": field, "message": "Must not be negative"}

    if not 0 <= data["OEE_Lag_1"] <= 100:
        return {"error": "Invalid OEE_Lag_1", "message": "Must be between 0 and 100"}
    if data["Maintenance_Flag_Lag_1"] not in (0, 1):
        return {"error": "Invalid maintenance flag", "message": "Must be 0 or 1"}
    return None


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    # App/model readiness only, not database connectivity.
    return jsonify(status="running", service="OEE Tomorrow Prediction API", model_loaded=True), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    error = validate_input(data)
    if error:
        return jsonify(error), 400

    try:
        input_data = pd.DataFrame([{field: data[field] for field in REQUIRED_FIELDS}])
        input_data = input_data[features]
        predicted_oee = round(float(model.predict(input_data)[0]), 2)
        if not math.isfinite(predicted_oee):
            raise ValueError("Model returned non-finite prediction")
        logger.info("Predicted OEE: %s", predicted_oee)
    except Exception:
        logger.exception("Prediction processing failed")
        return jsonify(error="Prediction processing failed"), 500

    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        ensure_predictions_table(connection)
        cursor = connection.cursor()
        cursor.execute(
            """INSERT INTO predictions (
                machine_age, oee_lag_1, avail_hrs_lag_1, mc_running_hrs_lag_1,
                total_dt_lag_1, prod_qty_lag_1, rej_qty_lag_1,
                maintenance_flag_lag_1, predicted_oee
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                data["Machine_Age"], data["OEE_Lag_1"], data["Avail_Hrs_Lag_1"],
                data["MC_Running_Hrs_Lag_1"], data["Total_DT_Lag_1"],
                data["Prod_Qty_Lag_1"], data["Rej_Qty_Lag_1"],
                data["Maintenance_Flag_Lag_1"], predicted_oee,
            ),
        )
        connection.commit()
        prediction_id = cursor.lastrowid
        logger.info("Prediction saved to Azure MySQL; ID: %s", prediction_id)
        return jsonify(
            message="Prediction successful and saved to Azure MySQL",
            prediction_id=prediction_id,
            predicted_oee=predicted_oee,
        ), 200
    except (mysql.connector.Error, OSError, KeyError, ValueError):
        if connection is not None:
            try:
                connection.rollback()
            except mysql.connector.Error:
                logger.warning("Could not roll back failed database operation")
        logger.exception("Azure MySQL operation failed")
        return jsonify(error="Database operation failed"), 503
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


@app.route("/predictions", methods=["GET"])
def get_predictions():
    # IMPORTANT: Add authentication before exposing this endpoint publicly.
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        ensure_predictions_table(connection)
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC, id DESC LIMIT 100")
        records = cursor.fetchall()
        for record in records:
            if isinstance(record.get("created_at"), datetime):
                record["created_at"] = record["created_at"].isoformat()
        return jsonify(records), 200
    except (mysql.connector.Error, OSError, KeyError, ValueError):
        logger.exception("Unable to retrieve Azure MySQL predictions")
        return jsonify(error="Unable to fetch predictions"), 503
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
