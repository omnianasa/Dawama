import streamlit as st
import torch
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
#load models with caching to avoid reloading on every interaction
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from model.FittedDawamaOFormer import FittedDawamaOFormer
    from model.OFormerDawama import OFormerDawama

    super_model = FittedDawamaOFormer(num_sensors=8, d_model=64, nhead=4).to(device)
    super_model.load_state_dict(torch.load("fitted_oformer_dawama_stable.pth", map_location=device))
    super_model.eval()

    standard_model = OFormerDawama(num_sensors=8, d_model=64, nhead=4).to(device)
    standard_model.load_state_dict(torch.load("oformer_dawama_model.pth", map_location=device))
    standard_model.eval()

    return device, super_model, standard_model

device, super_model, standard_model = load_models()


# ground truth velocity function (simulates real physics with some noise and environment effects)
def ground_truth_velocity(sensors, queries, wind_on):
    """
    Simulates the real physical velocity based on current sensor readings,
    query points, and wind condition.
    """
    # Base physics: 40 m/s nominal + contributions from sensors and queries
    vel = 40.0
    vel += 2.5 * np.mean(sensors)          # avg sensor effect
    vel += 1.2 * queries[0] + 0.8 * queries[1]
    
    if wind_on:
        vel += np.random.normal(2.0, 0.5)   # wind adds extra thrust/drag
    return np.clip(vel, 15.0, 70.0)

# Function to generate telemetry data based on environment and wind conditions
def generate_telemetry(env, wind_on):
    # Base random sensors (8 values) and queries (2 values)
    sensors = np.random.uniform(-0.2, 0.2, 8)
    queries = np.random.uniform(0.0, 0.8, 2)
    if env == 'noise':
        sensors += np.random.uniform(-0.5, 0.5, 8)
    elif env == 'failure':
        sensors[2] = 0.0
        sensors[6] = 0.0
    elif env == 'oob':
        queries = queries * 1.5 + 0.5   # push values > 1.0
    
    if wind_on:
        sensors = sensors * 2.5 + np.random.uniform(-0.4, 0.4, 8)
        queries[0] += 0.3
    
    # Clip and round for numerical stability
    sensors = np.clip(sensors, -1.5, 1.5)
    queries = np.clip(queries, 0.0, 2.0)
    return sensors.astype(np.float32), queries.astype(np.float32)

#ui
st.set_page_config(layout="wide", page_title="DAWAMA Flight Simulator")
st.title("🚁 DAWAMA UAV FLIGHT SIMULATOR with Ground Truth")

with st.sidebar:
    st.header("Flight Control")
    active_model = st.radio("Active Airflow Engine", ["Super OFormer (Stateful)", "Standard OFormer"], index=0)
    model_key = "super" if "Super" in active_model else "standard"
    
    st.subheader("Atmospheric Environment")
    env = st.selectbox("Environment", ["clean", "noise", "failure", "oob"], format_func=lambda x: {
        "clean": "1. Clean Weather",
        "noise": "2. Wind Noise (0.5)",
        "failure": "3. Sensor Failure",
        "oob": "4. Out of Bounds (1.5x)"
    }[x])
    
    wind_on = st.toggle("🌬️ Inject Sudden Wind", value=False)
    
    st.markdown("---")
    sim_speed = st.slider("Simulation Speed (steps/sec)", 1, 20, 8)
    start_btn = st.button("▶️ Start Telemetry Stream", type="primary")
    stop_btn = st.button("⏹️ Stop Stream")

if "running" not in st.session_state:
    st.session_state.running = False
if "data_log" not in st.session_state:
    st.session_state.data_log = []         
if "super_latent" not in st.session_state:
    st.session_state.super_latent = None
if "drone_tilt" not in st.session_state:
    st.session_state.drone_tilt = 0.0
if "drone_yoff" not in st.session_state:
    st.session_state.drone_yoff = 0.0
if "last_vel" not in st.session_state:
    st.session_state.last_vel = 40.0

if start_btn:
    st.session_state.running = True
    st.session_state.data_log = []
    st.session_state.super_latent = None
    st.session_state.last_vel = 40.0
if stop_btn:
    st.session_state.running = False

col1, col2 = st.columns([1, 2])

with col1:
    status_color = "green" if st.session_state.running else "red"
    st.metric("Neural Engine Telemetry", "Online - Streaming" if st.session_state.running else "Stopped")
    if st.session_state.data_log:
        current_pred = st.session_state.data_log[-1]["pred_vel"]
        current_true = st.session_state.data_log[-1]["true_vel"]
    else:
        current_pred = 40.0
        current_true = 40.0
    
    col_g1, col_g2 = st.columns(2)
    col_g1.metric("Predicted Vortex Airspeed", f"{current_pred:.3f} m/s")
    col_g2.metric("Ground Truth Velocity", f"{current_true:.3f} m/s", delta=f"{current_pred - current_true:.3f}")
    
    st.subheader("📡 LIVE UAV ORIENTATION")
    drone_container = st.empty()
    
    if wind_on:
        st.warning("🌬️ Turbulent wind active – expect instability")

with col2:
    st.subheader("Velocity over Time")
    chart_placeholder = st.empty()

def update_drone_figure(tilt, yoff, active_model, wind_on):
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    
    body_color = '#4EA8DE' if "super" in active_model else '#E63946'
    from matplotlib.transforms import Affine2D
    trans = Affine2D().rotate_deg(tilt).translate(0, yoff/50.0) + ax.transData
    
    # Rotors
    ax.plot([-0.8, -0.2], [0.4, 0.4], color='#56CFE1', lw=3, transform=trans)
    ax.plot([0.2, 0.8], [0.4, 0.4], color='#56CFE1', lw=3, transform=trans)
    # Chassis
    ax.plot([-0.6, 0.6], [0, 0], color=body_color, lw=8, solid_capstyle='round', transform=trans)
    # Center dome
    circle = plt.Circle((0, -0.15), 0.2, color='#0B132B', ec='#56CFE1', lw=2, transform=trans)
    ax.add_patch(circle)
    
    if wind_on:
        ax.annotate("💨", xy=(1.2, 0.5), fontsize=20, transform=trans)
    
    return fig


if st.session_state.running:
    status_text = st.empty()
    interval = 1.0 / sim_speed
    for step in range(1000): 
        if not st.session_state.running:
            break
        
        sensors, queries = generate_telemetry(env, wind_on)
        sensors_t = torch.from_numpy(sensors).unsqueeze(0).to(device)
        queries_t = torch.from_numpy(queries).unsqueeze(0).to(device)
        
        true_vel = ground_truth_velocity(sensors, queries, wind_on)
        with torch.no_grad():
            if model_key == "super":
                pred, new_latent = super_model(sensors_t, queries_t, st.session_state.super_latent)
                st.session_state.super_latent = new_latent.detach() if new_latent is not None else None
            else:
                pred = standard_model(sensors_t, queries_t)
                st.session_state.super_latent = None   # reset stateful memory for standard
            pred_vel = pred.cpu().item()
        

        vel_change = pred_vel - st.session_state.last_vel
        st.session_state.last_vel = pred_vel

        tilt = vel_change * 25
        yoff = vel_change * -10
        if wind_on:
            tilt += np.random.uniform(-5, 5)
            yoff += np.random.uniform(-4, 4)
        tilt = np.clip(tilt, -20, 20)
        yoff = np.clip(yoff, -25, 25)
        st.session_state.drone_tilt = tilt
        st.session_state.drone_yoff = yoff
        
        #Log data
        now = datetime.now().strftime("%H:%M:%S")
        st.session_state.data_log.append({
            "time": now,
            "true_vel": true_vel,
            "pred_vel": pred_vel,
            "env": env,
            "wind": wind_on
        })

        if len(st.session_state.data_log) > 40:
            st.session_state.data_log.pop(0)
        

        df = pd.DataFrame(st.session_state.data_log)
        if not df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df.index, df["true_vel"], label="Ground Truth", color='gray', linestyle='--', linewidth=2)
            ax.plot(df.index, df["pred_vel"], label=f"{active_model} Prediction", color='#4EA8DE' if "super" in model_key else '#E63946', linewidth=2)
            ax.set_ylabel("Velocity (m/s)")
            ax.set_xlabel("Step (most recent 40)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            chart_placeholder.pyplot(fig)
            plt.close(fig)
        
        drone_fig = update_drone_figure(st.session_state.drone_tilt, st.session_state.drone_yoff, model_key, wind_on)
        with col1:
            drone_container.pyplot(drone_fig)
            plt.close(drone_fig)
        

        status_text.info(f"Step {len(st.session_state.data_log)} | Env: {env} | Wind: {'ON' if wind_on else 'OFF'} | True: {true_vel:.2f} | Pred: {pred_vel:.2f}")

        time.sleep(interval)
    
    st.session_state.running = False
    st.success("Simulation stopped.")
else:
    st.info("Press **Start Telemetry Stream** to begin real‑time predictions with ground truth.")

    if st.session_state.data_log:
        df = pd.DataFrame(st.session_state.data_log)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df.index, df["true_vel"], label="Ground Truth", color='gray', linestyle='--')
        ax.plot(df.index, df["pred_vel"], label=f"{active_model} Prediction", color='#4EA8DE' if "super" in model_key else '#E63946')
        ax.legend()
        st.pyplot(fig)