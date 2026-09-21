# EdgeCrop-Zero

Edge AI Smart Irrigation Node
Objective
An autonomous, offline smart irrigation system powered by Edge AI (TinyML). This architecture eliminates cloud computing latency and the need for continuous internet connectivity by executing a transpiled machine learning model directly on a micro-power edge device. The system evaluates real-time soil moisture and NPK levels to trigger immediate physical irrigation, while asynchronously transmitting operational telemetry to a centralized backend for long-term analytics.

System Architecture
The project is divided into three core components:

The Edge ML Pipeline (/edge_ml)

Data Pipeline: Synthetic dataset generation simulating realistic bounds for soil moisture, nitrogen, phosphorus, potassium, and ambient temperature.

Model: A constrained DecisionTreeClassifier (capped at max depth 4) optimized for low-power edge deployment.

Transpilation: The Python model is transpiled into a bare-metal C header file (model.h) using micromlgen, converting the algorithmic weights into highly efficient, zero-dependency if/else logic.

The Telemetry Hub (/receiver)

API: A local FastAPI REST API acting as the central ingestion hub.

Validation: Pydantic models strictly enforce the schema of incoming JSON payloads from field nodes.

Database: SQLAlchemy ORM integrated with an SQLite database to permanently log node telemetry (moisture, temperature, and pump status) for historical crop analysis.

The Physical Node (Hardware Target)

Microcontroller: ESP32-WROOM-32.

Sensors: Capacitive soil moisture probes.

Execution: The ESP32 flashes the transpiled model.h C code, reads direct analog voltages, executes the irrigation decision offline, and uses onboard Wi-Fi to ping the FastAPI telemetry endpoint.

Tech Stack
Machine Learning: Python, scikit-learn, pandas, micromlgen

Backend: Python, FastAPI, uvicorn, SQLAlchemy, SQLite, Pydantic

Edge Firmware (Pending): C/C++
