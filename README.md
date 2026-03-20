# WindGuard AI — Flask Production Dashboard

Real-time wind turbine gearbox fault detection system with WebSocket streaming.

## Setup

```bash
pip install -r requirements.txt
```

If you have a trained model, copy it to the `models/` directory:
- `models/best_model.pkl`
- `models/model_metadata.json`
- `models/scaler.pkl`

To train the model from the original data:
```bash
python train_model.py
```

## Run

```bash
python app.py
```

Then open http://localhost:5000

## Pages

| URL | Description |
|-----|-------------|
| `/` | Home landing page |
| `/dashboard` | Fleet-wide real-time dashboard |
| `/monitoring` | Per-turbine live sensor monitoring |
| `/prediction` | ML fault prediction (demo/manual/CSV) |
| `/system` | System info, model metrics, team |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | System status summary |
| `/api/turbines` | GET | All turbine data |
| `/api/alerts` | GET | Recent fault alerts |
| `/api/predict/demo` | POST | Demo signal prediction |
| `/api/predict/manual` | POST | Manual parameter prediction |
| `/api/predict/csv` | POST | CSV file prediction |

## WebSocket Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `sensor_update` | Server→Client | Live sensor readings (every 2s) |
| `initial_state` | Server→Client | Initial turbine state on connect |
| `request_prediction` | Client→Server | Request ML prediction for turbine |
| `prediction_result` | Server→Client | ML prediction result |

## Features

- **Real-time WebSocket** streaming of 9-turbine sensor data
- **Dark/Light mode** toggle (persisted in localStorage)
- **Toast notifications** for fault events
- **Alert banner** for active faults
- **Responsive** Bootstrap 5 layout
- **Chart.js** line charts, bar charts, radar, donut charts
- **Demo mode** — runs ML inference on synthetic signals
- **CSV upload** — analyzes real vibration data files
- **Manual sliders** — adjust sensor parameters for instant prediction
