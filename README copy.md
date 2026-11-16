# Food Delivery Time Predictor (Flask)

## Folder layout
```
food_delivery_app/
├── app.py
├── requirements.txt
├── templates/index.html
└── model/
    ├── standard_scaler.pkl   (included)
    └── xgb_best.pkl          <-- YOU MUST ADD THIS
```

## Setup
1. In your notebook, run `joblib.dump(xgb_best, "xgb_best.pkl")` (you already do) and copy that file into `model/`.
2. Install and run:
```
pip install -r requirements.txt
python app.py
```
3. Open http://127.0.0.1:5000

## Notes
- Use the same scikit-learn / xgboost versions as your notebook (the scaler was saved with scikit-learn 1.6.1). Check with `pip show xgboost scikit-learn`.
- The form reproduces the notebook's preprocessing exactly: Haversine distance, order hour, traffic mapped to 0-3, one-hot columns with the first category dropped, then `scaler.transform`, then `model.predict`.
