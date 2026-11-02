// Edge Health Monitor Phone Sensors Client

let ws = null;
let sensorsRunning = false;
let audioContext = null;
let micStream = null;
let wakeLock = null;

let lastAccelTime = performance.now();
let accelCount = 0;
let lastGyroTime = performance.now();
let gyroCount = 0;

const toggleBtn = document.getElementById('toggle-sensors-btn');
const wsStatus = document.getElementById('ws-status');
const accXEl = document.getElementById('acc-x');
const accYEl = document.getElementById('acc-y');
const accZEl = document.getElementById('acc-z');
const accMagEl = document.getElementById('acc-mag');
const accelRateEl = document.getElementById('accel-rate');

const gyroXEl = document.getElementById('gyro-x');
const gyroYEl = document.getElementById('gyro-y');
const gyroZEl = document.getElementById('gyro-z');
const gyroRateEl = document.getElementById('gyro-rate');

const micStatusEl = document.getElementById('mic-status');
const audioLevelBar = document.getElementById('audio-level-bar');

let currentIMU = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];

function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/sensors`;
  
  ws = new WebSocket(wsUrl);
  
  ws.onopen = () => {
    wsStatus.innerHTML = '<span class="status-dot ok"></span>Connected';
  };
  
  ws.onclose = () => {
    wsStatus.innerHTML = '<span class="status-dot warning"></span>Disconnected';
    setTimeout(connectWebSocket, 2000);
  };
  
  ws.onerror = () => {
    wsStatus.innerHTML = '<span class="status-dot alert"></span>Error';
  };
}

async function startSensors() {
  // 1. Request iOS permission if required
  if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
    try {
      const permission = await DeviceMotionEvent.requestPermission();
      if (permission !== 'granted') {
        alert('Permission for device motion was denied.');
        return;
      }
    } catch (e) {
      console.warn('DeviceMotion permission error:', e);
    }
  }

  // 2. Request Wake Lock
  if ('wakeLock' in navigator) {
    try {
      wakeLock = await navigator.wakeLock.request('screen');
    } catch (e) {
      console.warn('Wake Lock error:', e);
    }
  }

  // 3. Start Device Motion Event
  window.addEventListener('devicemotion', handleMotionEvent);

  // 4. Start Microphone via Web Audio
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1
      }
    });

    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(micStream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);

    processor.onaudioprocess = (e) => {
      if (!sensorsRunning) return;
      const inputData = e.inputBuffer.getChannelData(0);
      
      // Compute RMS for level meter
      let sum = 0;
      for (let i = 0; i < inputData.length; i++) {
        sum += inputData[i] * inputData[i];
      }
      const rms = Math.sqrt(sum / inputData.length);
      const levelPct = Math.min(100, Math.round(rms * 400));
      audioLevelBar.style.width = `${levelPct}%`;

      // Convert float32 to PCM16
      const pcm16 = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        const s = Math.max(-1, Math.min(1, inputData[i]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }

      // Send audio chunk if WebSocket is open
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          source: 'phone',
          type: 'audio',
          timestamp: Date.now() / 1000.0,
          audio_pcm16: Array.from(pcm16.slice(0, 160)) // sample down to lightweight chunk
        }));
      }
    };

    source.connect(processor);
    processor.connect(audioContext.destination);
    micStatusEl.textContent = 'Active (16 kHz)';
  } catch (err) {
    console.warn('Microphone access denied or error:', err);
    micStatusEl.textContent = 'Mic unavailable';
  }

  sensorsRunning = true;
  toggleBtn.textContent = 'Stop sensors';
  toggleBtn.className = 'btn btn-secondary big-btn';

  // Transmit IMU stream at 50 Hz interval (20ms)
  startIMUSender();
}

function stopSensors() {
  sensorsRunning = false;
  window.removeEventListener('devicemotion', handleMotionEvent);
  
  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (wakeLock) {
    wakeLock.release().catch(() => {});
    wakeLock = null;
  }

  toggleBtn.textContent = 'Start sensors';
  toggleBtn.className = 'btn btn-primary big-btn';
  micStatusEl.textContent = 'Inactive';
  audioLevelBar.style.width = '0%';
}

function handleMotionEvent(event) {
  if (!sensorsRunning) return;

  // Accelerometer: accelerationIncludingGravity (m/s^2)
  const acc = event.accelerationIncludingGravity || event.acceleration;
  if (acc) {
    // Convert m/s^2 to g
    const ax = (acc.x || 0) / 9.80665;
    const ay = (acc.y || 0) / 9.80665;
    const az = (acc.z || 0) / 9.80665;
    const mag = Math.sqrt(ax * ax + ay * ay + az * az);

    accXEl.textContent = ax.toFixed(2);
    accYEl.textContent = ay.toFixed(2);
    accZEl.textContent = az.toFixed(2);
    accMagEl.textContent = mag.toFixed(2);

    currentIMU[0] = ax;
    currentIMU[1] = ay;
    currentIMU[2] = az;

    accelCount++;
    const now = performance.now();
    if (now - lastAccelTime >= 1000) {
      accelRateEl.textContent = `${accelCount} Hz`;
      accelCount = 0;
      lastAccelTime = now;
    }
  }

  // Gyroscope: rotationRate (deg/s -> rad/s)
  const gyro = event.rotationRate;
  if (gyro) {
    const degToRad = Math.PI / 180.0;
    const gx = (gyro.beta || 0) * degToRad;
    const gy = (gyro.gamma || 0) * degToRad;
    const gz = (gyro.alpha || 0) * degToRad;

    gyroXEl.textContent = gx.toFixed(2);
    gyroYEl.textContent = gy.toFixed(2);
    gyroZEl.textContent = gz.toFixed(2);

    currentIMU[3] = gx;
    currentIMU[4] = gy;
    currentIMU[5] = gz;

    gyroCount++;
    const now = performance.now();
    if (now - lastGyroTime >= 1000) {
      gyroRateEl.textContent = `${gyroCount} Hz`;
      gyroCount = 0;
      lastGyroTime = now;
    }
  }
}

let imuInterval = null;
function startIMUSender() {
  if (imuInterval) clearInterval(imuInterval);
  imuInterval = setInterval(() => {
    if (!sensorsRunning) {
      clearInterval(imuInterval);
      return;
    }
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        source: 'phone',
        type: 'imu',
        timestamp: Date.now() / 1000.0,
        imu: currentIMU
      }));
    }
  }, 20); // 50 Hz
}

toggleBtn.addEventListener('click', () => {
  if (sensorsRunning) {
    stopSensors();
  } else {
    startSensors();
  }
});

// Auto-connect on page load
connectWebSocket();
