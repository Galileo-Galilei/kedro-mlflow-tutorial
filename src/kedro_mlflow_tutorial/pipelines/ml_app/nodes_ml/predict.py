import xgboost as xgb


def predict_with_xgboost(model, data):
    xgb_data = xgb.DMatrix(data)
    return model.predict(xgb_data)


def decode_predictions(encoder, data):
    predictions = encoder.inverse_transform((data > 0.5) * 1)
    return predictions
