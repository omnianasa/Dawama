from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import json
from model.FittedDawamaOFormer import FittedDawamaOFormer

app = FastAPI(title="Project Dawama: Real-Time Aerodynamic Inference Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FittedDawamaOFormer(num_sensors=8, d_model=64, nhead=4).to(device)
model.load_state_dict(torch.load("fitted_oformer_dawama_stable.pth", map_location=device))
model.eval()

@app.get("/health")
def health_check():
    return {"status": "online", "device": str(device)}

@app.websocket("/stream/telemetry")
async def telemetry_stream(websocket: WebSocket):
    await websocket.accept()
    print("UAV Telemetry Connection Established.")
    super_latent = None
    
    try:
        while True:

            data = await websocket.receive_text()
            payload = json.loads(data)
            

            sensors = np.array(payload["sensors"], dtype=np.float32).reshape(1, 8)
            queries = np.array(payload["queries"], dtype=np.float32).reshape(1, 2)

            sensors_tensor = torch.from_numpy(sensors).to(device)
            queries_tensor = torch.from_numpy(queries).to(device)

            with torch.no_grad():
                preds, updated_kv = model(sensors_tensor, queries_tensor, super_latent)
                super_latent = updated_kv.detach() 
                
            predicted_velocity = preds.cpu().item()

            response_payload = {
                "status": "stabilized",
                "predicted_velocity": predicted_velocity,
                "sensor_health": "optimal" if np.min(sensors) > -2.0 else "degraded"
            }
            await websocket.send_text(json.dumps(response_payload))
            
    except WebSocketDisconnect:
        print("UAV Connection Closed. Resetting Flight Latent Memory.")
        super_latent = None