"""Python SDK"""
import requests

class TuningClient:
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000"):
        self.api_key = api_key
        self.base_url = base_url
        self.tunings = TuningAPI(self)

class TuningAPI:
    def __init__(self, client):
        self.client = client

    def create(self, **kwargs):
        response = requests.post(
            f"{self.client.base_url}/v1/tunings.create",
            json=kwargs,
            headers={"Authorization": f"Bearer {self.client.api_key}"}
        )
        return response.json()

    def get(self, job_id: str):
        response = requests.get(
            f"{self.client.base_url}/v1/tunings.get/{job_id}"
        )
        return response.json()
