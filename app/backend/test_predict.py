"""
Script de teste para validar endpoint /predict
"""
import requests
import json

API_URL = "http://localhost:8000"

# Payload de teste
payload = {
    "month": 6,
    "day": 15,
    "day_of_week": 3,
    "hour": 14,
    "distance": 1500,
    "scheduled_time": 180,
    "airline_delay_rate": 0.15,
    "origin_delay_rate": 0.18,
    "dest_delay_rate": 0.20,
    "route_delay_rate": 0.17,
    "origin_latitude": 33.9425,
    "origin_longitude": -118.4081,
    "dest_latitude": 40.7769,
    "dest_longitude": -73.8740,
    "origin_airport": 150,
    "destination_airport": 200,
    "origin_state": 5,
    "dest_state": 32,
    "is_weekend": 0,
    "is_morning_rush": 0,
    "is_evening_rush": 1,
    "is_domestic": 1,
    "airline_DL": 1,
    "airline_AS": 0,
    "airline_B6": 0,
    "airline_EV": 0,
    "airline_F9": 0,
    "airline_HA": 0,
    "airline_MQ": 0,
    "airline_NK": 0,
    "airline_OO": 0,
    "airline_UA": 0,
    "airline_US": 0,
    "airline_VX": 0,
    "airline_WN": 0,
    "period_of_day_Manha": 0,
    "period_of_day_Tarde": 1,
    "period_of_day_Noite": 0,
    "season_Outono": 0,
    "season_Primavera": 0,
    "season_Verao": 1
}

try:
    print("Testando endpoint /health...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

    print("Testando endpoint /predict...")
    response = requests.post(
        f"{API_URL}/predict",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        print(f"Success! Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")

except Exception as e:
    print(f"Erro: {e}")
