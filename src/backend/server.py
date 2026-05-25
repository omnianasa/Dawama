from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import json
from model.FittedDawamaOFormer import FittedDawamaOFormer 
from model.OFormerDawama import OFormerDawama  

app = FastAPI(title="Project Dawama")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
super_model = FittedDawamaOFormer(num_sensors=8, d_model=64, nhead=4).to(device)
super_model.load_state_dict(torch.load("fitted_oformer_dawama_stable.pth", map_location=device))
super_model.eval()

standard_model = OFormerDawama(num_sensors=8, d_model=64, nhead=4).to(device)
standard_model.load_state_dict(torch.load("oformer_dawama_model.pth", map_location=device))
standard_model.eval()

@app.websocket("/stream/telemetry")
async def telemetry_stream(websocket: WebSocket):
    await websocket.accept()
    print("Dual-Model Telemetry Stream Channel Opened.")
    
    super_latent = None
    
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
        
            sensors = np.array(payload["sensors"], dtype=np.float32).reshape(1, 8)
            queries = np.array(payload["queries"], dtype=np.float32).reshape(1, 2)
            active_model = payload.get("active_model", "super") 
            
            sensors_tensor = torch.from_numpy(sensors).to(device)
            queries_tensor = torch.from_numpy(queries).to(device)
            
            
            predicted_velocity = 40.0 
            
            try:
                with torch.no_grad():
                    if active_model == "super":
                        preds, updated_kv = super_model(sensors_tensor, queries_tensor, super_latent)
                        if updated_kv is not None:
                            super_latent = updated_kv.detach()
                        predicted_velocity = preds.cpu().item()
                    else:
                        preds = standard_model(sensors_tensor, queries_tensor)
                        predicted_velocity = preds.cpu().item()
                        super_latent = None  
            except Exception as e:
                print(print(f"Inference Error: {e}"))
                pass

            await websocket.send_text(json.dumps({
                "predicted_velocity": predicted_velocity
            }))
            
    except WebSocketDisconnect:
        print("Flight Channel Closed. Resetting Memories.")
        super_latent = None