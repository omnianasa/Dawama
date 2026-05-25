import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Gauge, Cpu, Wind, Activity } from 'lucide-react';

export default function App() {
  const [dataStream, setDataStream] = useState([]);
  const [currentVelocity, setCurrentVelocity] = useState(0);
  const [systemStatus, setSystemStatus] = useState("Connecting to Neural Server...");
  const [environment, setEnvironment] = useState('clean');
  const [isWindOn, setIsWindOn] = useState(false);
  const [activeModel, setActiveModel] = useState('super');
  const [droneTilt, setDroneTilt] = useState(0);
  const [droneYOffset, setDroneYOffset] = useState(0);

  const socketRef = useRef(null);
  const droneStateRef = useRef({ environment: 'clean', isWindOn: false, activeModel: 'super' });


  const lastVelocityRef = useRef(40);

  useEffect(() => {
    droneStateRef.current = { environment, isWindOn, activeModel };
  }, [environment, isWindOn, activeModel]);

  useEffect(() => {
    socketRef.current = new WebSocket("ws://localhost:8000/stream/telemetry");

    socketRef.current.onopen = () => {
      setSystemStatus("Online - Connected to Engine");

      const interval = setInterval(() => {
        if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) return;

        const state = droneStateRef.current;
        let baseSensors = Array.from({ length: 8 }, () => Math.random() * 0.4 - 0.2);
        let mockQueries = [Math.random() * 0.8, Math.random() * 0.8];

        if (state.environment === 'noise') {
          baseSensors = baseSensors.map(v => v + (Math.random() * 1.0 - 0.5));
        } else if (state.environment === 'failure') {
          baseSensors[2] = 0.0;
          baseSensors[6] = 0.0;
        } else if (state.environment === 'oob') {
          mockQueries = [1.5 + Math.random() * 0.5, 1.5 + Math.random() * 0.5];
        }

        if (state.isWindOn) {
          baseSensors = baseSensors.map(v => v * 2.5 + (Math.random() * 0.8 - 0.4));
          mockQueries[0] += 0.3;
        }

        const payload = {
          sensors: baseSensors.map(v => parseFloat(v.toFixed(4))),
          queries: mockQueries.map(v => parseFloat(v.toFixed(4))),
          active_model: state.activeModel
        };

        socketRef.current.send(JSON.stringify(payload));
      }, 150);

      return () => clearInterval(interval);
    };

    socketRef.current.onmessage = (event) => {
      const response = JSON.parse(event.data);
      const predictedVelocity = response.predicted_velocity;
      const state = droneStateRef.current;

      const velocityChange = predictedVelocity - lastVelocityRef.current;
      lastVelocityRef.current = predictedVelocity;

      let tiltFactor = velocityChange * 25;
      let bobbingFactor = velocityChange * -10;

      if (!state.isWindOn && Math.abs(velocityChange) < 0.01) {
        tiltFactor = 0;
        bobbingFactor = 0;
      }

      if (state.isWindOn) {
        tiltFactor += (Math.random() * 5 - 2.5);
        bobbingFactor += (Math.random() * 4 - 2);
      }

      setDroneTilt(Math.max(-20, Math.min(20, tiltFactor)));
      setDroneYOffset(Math.max(-25, Math.min(25, bobbingFactor)));
      setCurrentVelocity(predictedVelocity);

      setDataStream((prev) => {
        const updated = [...prev, { time: new Date().toLocaleTimeString().slice(-8), velocity: predictedVelocity }];
        if (updated.length > 40) updated.shift();
        return updated;
      });
    };

    socketRef.current.onclose = () => setSystemStatus("Offline - Connection Lost");
    socketRef.current.onerror = () => setSystemStatus("Error - Connection Failed");

    return () => socketRef.current && socketRef.current.close();
  }, []);

  return (
    <div style={{ backgroundColor: '#0B132B', color: '#FFFFFF', minHeight: '100vh', padding: '2rem', fontFamily: 'sans-serif' }}>

      {/* Top Header */}
      <div style={{ borderBottom: '2px solid #1C2541', paddingBottom: '1rem', marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1 style={{ color: '#4EA8DE', margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.6rem' }}>
          <Cpu /> DAWAMA UAV FLIGHT SIMULATOR v2.5
        </h1>
        <span style={{ backgroundColor: systemStatus.includes("Online") ? '#1B4332' : '#7209B7', color: '#FFF', padding: '0.5rem 1rem', borderRadius: '20px', fontSize: '0.85rem', fontWeight: 'bold' }}>
          {systemStatus}
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 3fr', gap: '2rem' }}>

        {/* Left Control Panel */}
        <div style={{ backgroundColor: '#1C2541', padding: '1.5rem', borderRadius: '12px', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <h3 style={{ margin: 0, color: '#56CFE1', borderBottom: '1px solid #0B132B', paddingBottom: '0.5rem' }}>Flight Control</h3>

          <div>
            <label style={{ fontSize: '0.85rem', color: '#6C757D', display: 'block', marginBottom: '0.5rem' }}>ACTIVE AIRFLOW ENGINE</label>
            <button onClick={() => setActiveModel('super')} style={{ width: '100%', padding: '0.75rem', marginBottom: '0.5rem', borderRadius: '6px', border: 'none', backgroundColor: activeModel === 'super' ? '#4EA8DE' : '#0B132B', color: '#FFF', fontWeight: 'bold', cursor: 'pointer', transition: '0.2s' }}>
              Super OFormer (Stateful)
            </button>
            <button onClick={() => setActiveModel('standard')} style={{ width: '100%', padding: '0.75rem', borderRadius: '6px', border: 'none', backgroundColor: activeModel === 'standard' ? '#E63946' : '#0B132B', color: '#FFF', fontWeight: 'bold', cursor: 'pointer', transition: '0.2s' }}>
              Standard OFormer
            </button>
          </div>

          <div>
            <label style={{ fontSize: '0.85rem', color: '#6C757D', display: 'block', marginBottom: '0.5rem' }}>ATMOSPHERIC ENVIRONMENT</label>
            {['clean', 'noise', 'failure', 'oob'].map((env) => (
              <button key={env} onClick={() => setEnvironment(env)} style={{ width: '100%', padding: '0.6rem', marginBottom: '0.4rem', borderRadius: '6px', border: 'none', backgroundColor: environment === env ? '#56CFE1' : '#0B132B', color: environment === env ? '#0B132B' : '#FFF', textTransform: 'uppercase', fontSize: '0.75rem', fontWeight: 'bold', cursor: 'pointer' }}>
                {env === 'clean' && '1. Clean Weather'}
                {env === 'noise' && '2. Wind Noise (0.5)'}
                {env === 'failure' && '3. Sensor Failure'}
                {env === 'oob' && '4. Out of Bounds (1.5x)'}
              </button>
            ))}
          </div>

          <div>
            <button onClick={() => setIsWindOn(!isWindOn)} style={{ width: '100%', padding: '1rem', borderRadius: '8px', border: 'none', backgroundColor: isWindOn ? '#FFB703' : '#1B4332', color: isWindOn ? '#000' : '#FFF', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', fontWeight: 'bold', cursor: 'pointer', boxShadow: isWindOn ? '0 0 15px #FFB703' : 'none', transition: '0.2s' }}>
              <Wind size={20} /> {isWindOn ? "TURBULENT WIND: ON" : "INJECT SUDDEN WIND"}
            </button>
          </div>
        </div>

        {/* Right Panel Grid */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

          {/* Indicators Layout (Clean Grid with 2 Columns Now) */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div style={{ backgroundColor: '#1C2541', padding: '1.2rem', borderRadius: '12px', display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <Gauge size={32} color="#4EA8DE" />
              <div>
                <div style={{ fontSize: '0.8rem', color: '#6C757D' }}>VORTEX AIRSPEED</div>
                <div style={{ fontSize: '1.4rem', fontWeight: 'bold', color: '#4EA8DE' }}>{currentVelocity.toFixed(4)} m/s</div>
              </div>
            </div>

            <div style={{ backgroundColor: '#1C2541', padding: '1.2rem', borderRadius: '12px', display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <Activity size={32} color={systemStatus.includes('Online') ? '#56CFE1' : '#E63946'} />
              <div>
                <div style={{ fontSize: '0.8rem', color: '#6C757D' }}>NEURAL ENGINE TELEMETRY</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 'bold', color: systemStatus.includes('Online') ? '#56CFE1' : '#E63946' }}>{systemStatus}</div>
              </div>
            </div>
          </div>

          {/* Live 2D Drone Physics Visualization Box */}
          <div style={{ backgroundColor: '#1C2541', padding: '1.5rem', borderRadius: '12px', height: '180px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', position: 'relative', overflow: 'hidden', border: isWindOn ? '1px dashed #FFB703' : '1px solid #1C2541' }}>
            <div style={{ position: 'absolute', top: '10px', left: '15px', fontSize: '0.8rem', color: '#56CFE1', fontWeight: 'bold' }}>📡 LIVE UAV ORIENTATION VECTOR</div>

            {isWindOn && <div style={{ position: 'absolute', width: '100%', height: '4px', backgroundColor: 'rgba(255, 183, 3, 0.2)', top: '50%', transform: 'translateY(-50%)' }} />}

            <div style={{
              transform: `translateY(${droneYOffset}px) rotate(${droneTilt}deg)`,
              transition: 'transform 0.25s ease-out',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              width: '240px'
            }}>
              {/* Rotors */}
              <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', paddingBottom: '4px' }}>
                <div style={{ width: '40px', height: '4px', backgroundColor: '#56CFE1', borderRadius: '2px', animation: 'spin 0.05s linear infinite' }} />
                <div style={{ width: '40px', height: '4px', backgroundColor: '#56CFE1', borderRadius: '2px', animation: 'spin 0.05s linear infinite' }} />
              </div>

              {/* Chassis Body*/}
              <div style={{ width: '180px', height: '12px', backgroundColor: activeModel === 'super' ? '#4EA8DE' : '#E63946', borderRadius: '6px', position: 'relative', display: 'flex', justifyContent: 'center', alignItems: 'center', boxShadow: activeModel === 'super' ? '0 0 15px rgba(78,168,222,0.5)' : '0 0 15px rgba(230,57,70,0.5)', transition: 'background-color 0.3s' }}>
                <div style={{ width: '30px', height: '30px', backgroundColor: '#0B132B', border: '3px solid #56CFE1', borderRadius: '50%', position: 'absolute', bottom: '-8px' }} />
              </div>


              <div style={{ display: 'flex', justifyContent: 'space-between', width: '140px', height: '10px', borderLeft: '3px solid #6C757D', borderRight: '3px solid #6C757D', marginTop: '2px' }} />
            </div>
          </div>

          {/* Mathematical Chart Plot */}
          <div style={{ backgroundColor: '#1C2541', padding: '1.5rem', borderRadius: '12px', height: '320px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={dataStream}>
                <CartesianGrid strokeDasharray="3 3" stroke="#0B132B" />
                <XAxis dataKey="time" stroke="#6C757D" />
                <YAxis stroke="#6C757D" domain={['auto', 'auto']} />
                <Tooltip contentStyle={{ backgroundColor: '#0B132B', borderColor: '#4EA8DE' }} />
                <Line type="monotone" dataKey="velocity" stroke={activeModel === 'super' ? '#4EA8DE' : '#E63946'} strokeWidth={3} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

        </div>
      </div>

      <style>{`
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}