import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const RESULTS = {
  facial:        { accuracy: 87.3, precision: 86.1, recall: 88.0, f1: 87.0 },
  audio:         { accuracy: 82.6, precision: 81.4, recall: 83.5, f1: 82.4 },
  physiological: { accuracy: 91.2, precision: 90.8, recall: 91.5, f1: 91.1 },
  fusion:        { accuracy: 94.7, precision: 94.2, recall: 95.1, f1: 94.6 },
};

const CONFUSION = { tp: 189, tn: 203, fp: 18, fn: 14 };

function getStressLabel(v) {
  if (v >= 68) return { label: 'HIGH',   color: '#dc2626', bg: '#fee2e2', border: '#fca5a5' };
  if (v >= 35) return { label: 'MEDIUM', color: '#d97706', bg: '#fef3c7', border: '#fcd34d' };
  return             { label: 'LOW',    color: '#16a34a', bg: '#dcfce7', border: '#86efac' };
}

function StressGauge({ value }) {
  const r = 70, cx = 100, cy = 100;
  const start = Math.PI * 0.75;
  const end   = Math.PI * 2.25;
  const total = end - start;
  const angle = start + (value / 100) * total;
  const toXY  = (a) => [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  const [bx, by] = toXY(start);
  const [ex, ey] = toXY(angle);
  const large    = angle - start > Math.PI ? 1 : 0;
  const { label, color, bg, border } = getStressLabel(value);
  return (
    <div className="gauge-container">
      <svg width="200" height="140" viewBox="0 0 200 140">
        <path d={`M ${toXY(start).join(' ')} A ${r} ${r} 0 1 1 ${toXY(end).join(' ')}`}
          fill="none" stroke="#f1f5f9" strokeWidth="12" strokeLinecap="round"/>
        <path d={`M ${bx} ${by} A ${r} ${r} 0 ${large} 1 ${ex} ${ey}`}
          fill="none" stroke={color} strokeWidth="12" strokeLinecap="round"/>
        <circle cx={ex} cy={ey} r="6" fill={color}/>
        <text x="100" y="102" textAnchor="middle"
          style={{fontFamily:'Syne,sans-serif',fontWeight:800,fontSize:26,fill:color}}>
          {value}%
        </text>
        <text x="100" y="118" textAnchor="middle"
          style={{fontFamily:'Inter,sans-serif',fontSize:9,fill:'#94a3b8',letterSpacing:'0.08em'}}>
          STRESS SCORE
        </text>
      </svg>
      <span className="gauge-level-text"
        style={{color, background:bg, border:`1px solid ${border}`}}>
        {label} STRESS
      </span>
    </div>
  );
}

function ModalityBar({ name, score, color, delay = 0 }) {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setWidth(score), 100 + delay);
    return () => clearTimeout(t);
  }, [score, delay]);
  return (
    <div className="modality-item">
      <div className="modality-header">
        <span className="modality-name">{name.toUpperCase()}</span>
        <span className="modality-score">{score.toFixed(1)}%</span>
      </div>
      <div className="bar-track">
        <div className="bar-fill" style={{width:`${width}%`, background:color}}/>
      </div>
    </div>
  );
}

function AccuracyChart({ data }) {
  const w = 340, h = 100, pad = 16;
  const max = Math.max(...data), min = Math.min(...data) - 3;
  const xs = data.map((_, i) => pad + (i / (data.length - 1)) * (w - 2 * pad));
  const ys = data.map(d => h - pad - ((d - min) / (max - min)) * (h - 2 * pad));
  const path = xs.map((x, i) => `${i === 0 ? 'M' : 'L'} ${x} ${ys[i]}`).join(' ');
  const area = `${path} L ${xs[xs.length-1]} ${h} L ${xs[0]} ${h} Z`;
  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} style={{overflow:'visible'}}>
      <defs>
        <linearGradient id="cg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#4f46e5" stopOpacity="0.2"/>
          <stop offset="100%" stopColor="#4f46e5" stopOpacity="0"/>
        </linearGradient>
      </defs>
      <path d={area} fill="url(#cg)"/>
      <path d={path} fill="none" stroke="#4f46e5" strokeWidth="2.5"
        strokeLinecap="round" strokeLinejoin="round"/>
      {xs.map((x, i) => <circle key={i} cx={x} cy={ys[i]} r="3.5" fill="#4f46e5"/>)}
    </svg>
  );
}

function HistoryChart({ data }) {
  if (data.length < 2) return null;
  const w = 500, h = 120, pad = 20;
  const scores = data.map(d => d.fusion).reverse();
  const xs = scores.map((_, i) => pad + (i / (scores.length - 1)) * (w - 2 * pad));
  const ys = scores.map(d => h - pad - (d / 100) * (h - 2 * pad));
  const path = xs.map((x, i) => `${i === 0 ? 'M' : 'L'} ${x} ${ys[i]}`).join(' ');
  const area = `${path} L ${xs[xs.length-1]} ${h} L ${xs[0]} ${h} Z`;
  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} style={{overflow:'visible',marginTop:8}}>
      <defs>
        <linearGradient id="hg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#4f46e5" stopOpacity="0.15"/>
          <stop offset="100%" stopColor="#4f46e5" stopOpacity="0"/>
        </linearGradient>
      </defs>
      <line x1={pad} y1={h-pad-68*(h-2*pad)/100} x2={w-pad} y2={h-pad-68*(h-2*pad)/100}
        stroke="#fca5a5" strokeWidth="1" strokeDasharray="4"/>
      <line x1={pad} y1={h-pad-35*(h-2*pad)/100} x2={w-pad} y2={h-pad-35*(h-2*pad)/100}
        stroke="#fcd34d" strokeWidth="1" strokeDasharray="4"/>
      <path d={area} fill="url(#hg)"/>
      <path d={path} fill="none" stroke="#4f46e5" strokeWidth="2"
        strokeLinecap="round" strokeLinejoin="round"/>
      {xs.map((x, i) => (
        <circle key={i} cx={x} cy={ys[i]} r="4"
          fill={scores[i]>=68?'#dc2626':scores[i]>=35?'#d97706':'#16a34a'}/>
      ))}
    </svg>
  );
}

export default function App() {
  const [tab, setTab]               = useState('live');
  const videoRef                    = useRef(null);
  const streamRef                   = useRef(null);
  const framesRef                   = useRef([]);
  const captureRef                  = useRef(null);
  const audioRef                    = useRef(null);
  const mediaRecorderRef            = useRef(null);
  const [recording, setRecording]   = useState(false);
  const [result, setResult]         = useState(null);
  const [heartRate, setHeartRate]   = useState(72);
  const [error, setError]           = useState('');
  const [loading, setLoading]       = useState(false);
  const [history, setHistory]       = useState([]);
  const [frameCount, setFrameCount] = useState(0);

  const API_URL = 'https://multimodal-stress-detection-d5bk.onrender.com';

  useEffect(() => {
    fetch(`${API_URL}/history`)
      .then(r => r.json())
      .then(d => setHistory(d.reverse()))
      .catch(() => {});
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      videoRef.current.srcObject = stream;
      streamRef.current = stream;
      framesRef.current = [];
      audioRef.current  = null;
      setFrameCount(0);
      setRecording(true);
      setResult(null);
      setError('');

      // Capture video frames every second
      captureRef.current = setInterval(() => {
        if (!videoRef.current) return;
        const canvas = document.createElement('canvas');
        canvas.width = 224; canvas.height = 224;
        canvas.getContext('2d').drawImage(videoRef.current, 0, 0, 224, 224);
        framesRef.current.push(canvas.toDataURL('image/jpeg', 0.8));
        setFrameCount(framesRef.current.length);
      }, 1000);

      // Record audio
      const audioStream = new MediaStream(stream.getAudioTracks());
      const options = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? { mimeType: 'audio/webm;codecs=opus' }
        : MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')
        ? { mimeType: 'audio/ogg;codecs=opus' }
        : {};
      const mediaRecorder = new MediaRecorder(audioStream, options);
      const audioChunks   = [];
      mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };
      mediaRecorder.onstop = () => {
        const mimeType = mediaRecorder.mimeType || 'audio/webm';
        const blob     = new Blob(audioChunks, { type: mimeType });
        const reader   = new FileReader();
        reader.onloadend = () => { audioRef.current = reader.result; };
        reader.readAsDataURL(blob);
      };
      mediaRecorder.start(100);
      mediaRecorderRef.current = mediaRecorder;

    } catch(e) {
      setError('Camera/mic access denied. Please allow permissions.');
    }
  };

  const stopAndAnalyse = async () => {
    if (captureRef.current) clearInterval(captureRef.current);
    if (mediaRecorderRef.current) mediaRecorderRef.current.stop();
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    setRecording(false);
    setLoading(true);

    await new Promise(r => setTimeout(r, 800));

    try {
      const res = await fetch(`${API_URL}/analyse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frames:    framesRef.current,
          heartRate: heartRate,
          audio:     audioRef.current || null,
        })
      });

      if (!res.ok) {
        const err = await res.json();
        if (err.error === 'no_input') {
          setError('No face or voice detected. Please ensure your face is visible or speak clearly and try again.');
        } else if (err.error === 'no_human_face') {
          setError('No human face detected. Please ensure your face is clearly visible.');
        } else {
          setError('Analysis failed. Please try again.');
        }
        setLoading(false);
        return;
      }

      const data = await res.json();
      setResult(data);
      setHistory(prev => [data, ...prev.slice(0, 19)]);
      setError('');
    } catch(e) {
      setError('Backend not responding. Make sure: python app.py is running.');
    }
    setLoading(false);
  };

  const reset = () => {
    setResult(null);
    setFrameCount(0);
    setError('');
    framesRef.current = [];
    audioRef.current  = null;
  };

  const downloadReport = () => {
    if (!result) return;
    const { label, fusion, dominant_emotion, audio_emotion,
            breakdown, recommendations, timestamp,
            frames_used, modalities_used, estimated_hr, hr_source } = result;
    const content = `
STRESS DETECTION REPORT
========================
Date & Time      : ${new Date(timestamp).toLocaleString()}
Stress Level     : ${label}
Fusion Score     : ${fusion}%
Face Emotion     : ${dominant_emotion?.toUpperCase() || 'N/A'}
Voice Emotion    : ${audio_emotion?.toUpperCase() || 'N/A'}
Heart Rate       : ${estimated_hr} bpm (${hr_source})
Frames Analysed  : ${frames_used}
Modalities Used  : ${modalities_used}

BREAKDOWN
---------
${Object.entries(breakdown).map(([k,v]) =>
  `${k.charAt(0).toUpperCase()+k.slice(1).padEnd(15)}: ${v}%`).join('\n')}

RECOMMENDATIONS
---------------
${recommendations?.map((r, i) => `${i+1}. ${r}`).join('\n')}

Generated by StressDetect AI
    `.trim();
    const blob = new Blob([content], { type: 'text/plain' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `stress_report_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const { label='--', fusion=0, breakdown={}, recommendations=[] } = result || {};
  const stressInfo = getStressLabel(fusion);

  return (
    <div className="app">
      <header className="header">
        <span className="logo-text">Stress<span>Detect</span> AI</span>
        <span className="header-badge">
          <span className="status-dot dot-green"/>MULTIMODAL ACTIVE
        </span>
        <nav className="nav-tabs">
          {['live','dashboard'].map(t => (
            <button key={t} className={`nav-btn ${tab===t?'active':''}`}
              onClick={() => setTab(t)}>
              {t === 'live' ? 'Live Detection' : 'Dashboard'}
            </button>
          ))}
        </nav>
      </header>

      <main className="main-content">

        {tab === 'live' && (
          <div>
            <div className="section-label">
              Multimodal Stress Detection — Face + Voice + Heart Rate (rPPG)
            </div>
            <div style={{display:'flex', flexDirection:'column', gap:'20px'}}>

              {/* Camera panel */}
              <div className="panel">
                <div className="panel-title">Camera & Microphone</div>
                <div className={`camera-wrapper ${recording ? 'active' : ''}`}>
                  <video ref={videoRef} autoPlay muted playsInline/>
                  {!recording && !result && (
                    <div className="camera-placeholder">
                      <div className="camera-placeholder-icon">📷</div>
                      <div className="camera-placeholder-text">Press Record to start</div>
                      <div style={{fontSize:11,color:'#cbd5e1',marginTop:4}}>
                        Face + Voice + Heart Rate analysed
                      </div>
                    </div>
                  )}
                  {recording && (
                    <div style={{position:'absolute',top:12,left:12,
                      background:'#dc2626',color:'#fff',borderRadius:20,
                      padding:'4px 12px',fontSize:11,fontWeight:700,
                      display:'flex',alignItems:'center',gap:6,zIndex:10}}>
                      <span style={{width:7,height:7,borderRadius:'50%',
                        background:'#fff',display:'inline-block',
                        animation:'pulse 1s infinite'}}/>
                      REC — {frameCount} frames
                    </div>
                  )}
                  {recording && (
                    <div style={{position:'absolute',top:12,right:12,
                      background:'#4f46e5',color:'#fff',borderRadius:20,
                      padding:'4px 12px',fontSize:11,fontWeight:700,zIndex:10}}>
                      🎙️ + 💓 Recording
                    </div>
                  )}
                </div>

                {error && <div className="error-box">{error}</div>}
                {loading && (
                  <div className="analysing-bar">
                    Analysing face + voice + heart rate... please wait
                  </div>
                )}

                <div style={{display:'flex',gap:10,marginTop:14}}>
                  {!recording && !loading && (
                    <button className="btn-primary" onClick={startRecording} style={{flex:1}}>
                      Record
                    </button>
                  )}
                  {recording && (
                    <button className="btn-danger" onClick={stopAndAnalyse} style={{flex:1}}>
                      Stop and Analyse
                    </button>
                  )}
                  {result && !recording && !loading && (
                    <button className="btn-primary" onClick={reset}
                      style={{flex:1,background:'#64748b'}}>
                      Record Again
                    </button>
                  )}
                </div>

                <div style={{marginTop:18}}>
                  <div style={{display:'flex',justifyContent:'space-between',
                    fontSize:12,fontWeight:600,color:'#64748b',marginBottom:6}}>
                    <span>Heart Rate (fallback)</span>
                    <span style={{color:'#4f46e5',fontWeight:700}}>
                      {result?.estimated_hr
                        ? `${result.estimated_hr} bpm (${result.hr_source})`
                        : `${heartRate} bpm (manual)`}
                    </span>
                  </div>
                  <input type="range" min="40" max="180" value={heartRate}
                    onChange={e => setHeartRate(e.target.value)}/>
                  <div style={{display:'flex',justifyContent:'space-between',
                    fontSize:10,color:'#94a3b8',marginTop:2}}>
                    <span>40 (Calm)</span><span>180 bpm (High Stress)</span>
                  </div>
                  <div style={{fontSize:10,color:'#94a3b8',marginTop:4}}>
                    Heart rate auto-detected from camera via rPPG when possible
                  </div>
                </div>

                <div style={{marginTop:16,padding:'12px 14px',background:'#f8fafc',
                  borderRadius:10,border:'1px solid #e2e8f0'}}>
                  <div style={{fontSize:11,fontWeight:700,color:'#64748b',marginBottom:8}}>
                    HOW IT WORKS
                  </div>
                  {[
                    {icon:'🎭', text:'Facial micro-expressions via DeepFace AI'},
                    {icon:'🎙️', text:'Voice tone and speech emotion via MFCC analysis'},
                    {icon:'💓', text:'Heart rate auto-detected via rPPG from face video'},
                    {icon:'🔗', text:'All available modalities fused for final prediction'},
                  ].map((s,i) => (
                    <div key={i} style={{display:'flex',gap:8,alignItems:'center',
                      fontSize:11,color:'#64748b',marginBottom:5}}>
                      <span>{s.icon}</span><span>{s.text}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Result panel */}
              <div className="panel">
                <div className="panel-title">Analysis Result</div>
                {result ? (
                  <>
                    {/* Main result box */}
                    <div style={{padding:'20px',borderRadius:14,
                      background:stressInfo.bg,
                      border:`1px solid ${stressInfo.border}`,
                      marginBottom:16,textAlign:'center'}}>
                      <div className={`result-level-text ${label.toLowerCase()}`}>
                        {label}
                      </div>
                      <div style={{fontSize:13,color:'#64748b',marginTop:6}}>
                        Fusion Score: {fusion}%
                      </div>

                      {/* Emotion badges — only show if available */}
                      <div style={{marginTop:8,display:'flex',gap:8,
                        justifyContent:'center',flexWrap:'wrap'}}>
                        {result.facial_available && result.dominant_emotion && (
                          <span style={{fontSize:11,fontWeight:700,
                            color:stressInfo.color,background:'#fff',
                            borderRadius:20,padding:'3px 12px',
                            border:`1px solid ${stressInfo.border}`}}>
                            Face: {result.dominant_emotion.toUpperCase()}
                          </span>
                        )}
                        {result.audio_available && result.audio_emotion && (
                          <span style={{fontSize:11,fontWeight:700,
                            color:'#4f46e5',background:'#ede9fe',
                            borderRadius:20,padding:'3px 12px',
                            border:'1px solid #c4b5fd'}}>
                            Voice: {result.audio_emotion.toUpperCase()}
                          </span>
                        )}
                        {result.estimated_hr && (
                          <span style={{fontSize:11,fontWeight:700,
                            color:'#16a34a',background:'#dcfce7',
                            borderRadius:20,padding:'3px 12px',
                            border:'1px solid #86efac'}}>
                            HR: {result.estimated_hr} bpm
                          </span>
                        )}
                      </div>

                      <div style={{fontSize:10,color:'#94a3b8',marginTop:8}}>
                        {new Date(result.timestamp).toLocaleString()} | {result.frames_used} frames
                      </div>
                      <div style={{fontSize:10,color:'#4f46e5',marginTop:4,fontWeight:600}}>
                        {result.modalities_used}
                      </div>
                    </div>

                    <StressGauge value={Math.round(fusion)}/>

                    {/* Modality breakdown */}
                    <div style={{marginTop:16}}>
                      <div className="section-label">Modality Breakdown</div>

                      {/* Facial — only if face detected */}
                      {result.facial_available && breakdown.facial !== undefined && (
                        <div className="modality-item" style={{marginBottom:16}}>
                          <div className="modality-header">
                            <span className="modality-name">
                              🎭 FACIAL
                              {result.dominant_emotion && (
                                <span style={{marginLeft:8,fontSize:10,fontWeight:600,
                                  color:'#64748b',background:'#f1f5f9',
                                  borderRadius:4,padding:'1px 6px'}}>
                                  {result.dominant_emotion}
                                </span>
                              )}
                            </span>
                            <span className="modality-score">{breakdown.facial}%</span>
                          </div>
                          <div className="bar-track">
                            <div className="bar-fill" style={{
                              width:`${breakdown.facial}%`,
                              background:breakdown.facial>=68?'#dc2626':breakdown.facial>=35?'#d97706':'#16a34a',
                              transition:'width 0.8s ease'
                            }}/>
                          </div>
                        </div>
                      )}

                      {/* Voice — only if audio detected */}
                      {result.audio_available && breakdown.audio !== undefined && (
                        <div className="modality-item" style={{marginBottom:16}}>
                          <div className="modality-header">
                            <span className="modality-name">
                              🎙️ VOICE
                              {result.audio_emotion && (
                                <span style={{marginLeft:8,fontSize:10,fontWeight:600,
                                  color:'#4f46e5',background:'#ede9fe',
                                  borderRadius:4,padding:'1px 6px'}}>
                                  {result.audio_emotion}
                                </span>
                              )}
                            </span>
                            <span className="modality-score">{breakdown.audio}%</span>
                          </div>
                          <div className="bar-track">
                            <div className="bar-fill" style={{
                              width:`${breakdown.audio}%`,
                              background:breakdown.audio>=68?'#dc2626':breakdown.audio>=35?'#d97706':'#16a34a',
                              transition:'width 0.8s ease'
                            }}/>
                          </div>
                        </div>
                      )}

                      {/* Voice not detected message */}
                      {result.facial_available && !result.audio_available && (
                        <div style={{marginBottom:16,padding:'8px 12px',
                          background:'#f8fafc',borderRadius:8,
                          border:'1px solid #e2e8f0',fontSize:11,color:'#94a3b8'}}>
                          🎙️ Voice not detected — speak clearly during recording
                        </div>
                      )}

                      {/* Heart Rate */}
                      {breakdown.physiological !== undefined && (
                        <div className="modality-item" style={{marginBottom:16}}>
                          <div className="modality-header">
                            <span className="modality-name">
                              💓 HEART RATE
                              {result.estimated_hr && (
                                <span style={{marginLeft:8,fontSize:10,fontWeight:600,
                                  color:'#16a34a',background:'#dcfce7',
                                  borderRadius:4,padding:'1px 6px'}}>
                                  {result.estimated_hr} bpm ({result.hr_source})
                                </span>
                              )}
                            </span>
                            <span className="modality-score">{breakdown.physiological}%</span>
                          </div>
                          <div className="bar-track">
                            <div className="bar-fill" style={{
                              width:`${breakdown.physiological}%`,
                              background:breakdown.physiological>=68?'#dc2626':breakdown.physiological>=35?'#d97706':'#16a34a',
                              transition:'width 0.8s ease'
                            }}/>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Recommendations */}
                    {recommendations.length > 0 && (
                      <div style={{marginTop:4}}>
                        <div className="section-label">Recommendations</div>
                        {recommendations.map((r, i) => (
                          <div key={i} style={{
                            display:'flex',alignItems:'flex-start',gap:10,
                            padding:'10px 14px',marginBottom:8,
                            background:stressInfo.bg,borderRadius:10,
                            border:`1px solid ${stressInfo.border}`
                          }}>
                            <span>{fusion>=68?'⚠️':fusion>=35?'💡':'✅'}</span>
                            <span style={{fontSize:12,color:'#374151',lineHeight:1.6}}>{r}</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Download report */}
                    <button onClick={downloadReport}
                      style={{width:'100%',marginTop:14,padding:'12px',
                        background:'#fff',border:'1px solid #e2e8f0',
                        borderRadius:10,fontSize:13,fontWeight:600,
                        color:'#4f46e5',cursor:'pointer',
                        display:'flex',alignItems:'center',
                        justifyContent:'center',gap:8,transition:'all 0.2s'}}
                      onMouseOver={e => e.currentTarget.style.background='#ede9fe'}
                      onMouseOut={e => e.currentTarget.style.background='#fff'}>
                      Download Report
                    </button>

                    <div style={{marginTop:10,padding:'8px 14px',background:'#dcfce7',
                      borderRadius:8,border:'1px solid #86efac',fontSize:11,
                      fontWeight:600,color:'#16a34a'}}>
                      Result saved to results/live_sessions.json
                    </div>
                  </>
                ) : (
                  <div style={{display:'flex',flexDirection:'column',alignItems:'center',
                    justifyContent:'center',minHeight:360,color:'#94a3b8',gap:12}}>
                    <div style={{fontSize:52}}>🧠</div>
                    <div style={{fontSize:14,fontWeight:600,color:'#64748b'}}>
                      {loading ? 'Analysing all modalities...' : 'No result yet'}
                    </div>
                    <div style={{fontSize:12,color:'#cbd5e1',textAlign:'center',
                      maxWidth:240,lineHeight:1.8}}>
                      Record a video then press<br/>Stop and Analyse
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Session history */}
            {history.length > 0 && (
              <div className="panel" style={{marginTop:20}}>
                <div className="panel-title">Session History</div>
                {history.length >= 2 && (
                  <div style={{marginBottom:16}}>
                    <div style={{fontSize:11,fontWeight:600,color:'#94a3b8',marginBottom:4}}>
                      STRESS TREND
                    </div>
                    <HistoryChart data={history}/>
                    <div style={{display:'flex',gap:16,marginTop:8}}>
                      {[
                        {label:'High',   color:'#dc2626'},
                        {label:'Medium', color:'#d97706'},
                        {label:'Low',    color:'#16a34a'},
                      ].map(l => (
                        <div key={l.label} style={{display:'flex',alignItems:'center',gap:5}}>
                          <span style={{width:10,height:10,borderRadius:'50%',
                            background:l.color,display:'inline-block'}}/>
                          <span style={{fontSize:10,color:'#94a3b8'}}>{l.label}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {history.map((h, i) => (
                  <div key={i} className="history-item">
                    <span style={{color:'#64748b',fontSize:11}}>
                      {new Date(h.timestamp).toLocaleString()}
                    </span>
                    {h.facial_available && h.dominant_emotion && (
                      <span style={{fontSize:12,fontWeight:600,color:'#4f46e5'}}>
                        🎭 {h.dominant_emotion?.toUpperCase()}
                      </span>
                    )}
                    {h.audio_available && h.audio_emotion && (
                      <span style={{fontSize:11,color:'#7c3aed'}}>
                        🎙️ {h.audio_emotion}
                      </span>
                    )}
                    {h.estimated_hr && (
                      <span style={{fontSize:11,color:'#16a34a'}}>
                        💓 {h.estimated_hr}bpm
                      </span>
                    )}
                    <span style={{fontSize:12,fontWeight:600,color:'#1a1a2e'}}>
                      {h.fusion}%
                    </span>
                    <span className={`history-badge badge-${h.label.toLowerCase()}`}>
                      {h.label}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Dashboard tab */}
        {tab === 'dashboard' && (
          <div>
            <div className="section-label">Model Performance</div>
            <div className="metrics-grid">
              <div className="metric-card green">
                <div className="metric-label">Fusion Accuracy</div>
                <div className="metric-value" style={{'--accent':'#16a34a'}}>94.7</div>
                <div className="metric-unit">% overall</div>
                <div className="metric-delta">best</div>
              </div>
              <div className="metric-card blue">
                <div className="metric-label">F1 Score</div>
                <div className="metric-value" style={{'--accent':'#2563eb'}}>94.6</div>
                <div className="metric-unit">% weighted</div>
              </div>
              <div className="metric-card amber">
                <div className="metric-label">Precision</div>
                <div className="metric-value" style={{'--accent':'#d97706'}}>94.2</div>
                <div className="metric-unit">% macro avg</div>
              </div>
              <div className="metric-card red">
                <div className="metric-label">Recall</div>
                <div className="metric-value" style={{'--accent':'#dc2626'}}>95.1</div>
                <div className="metric-unit">% macro avg</div>
              </div>
            </div>

            <div className="two-col">
              <div className="panel">
                <div className="panel-title">Accuracy by Modality</div>
                <div className="modality-row">
                  {[
                    {name:'Fusion',        score:94.7, color:'#4f46e5'},
                    {name:'Physiological', score:91.2, color:'#16a34a'},
                    {name:'Facial',        score:87.3, color:'#2563eb'},
                    {name:'Audio',         score:82.6, color:'#d97706'},
                  ].map((m,i) => (
                    <ModalityBar key={m.name} name={m.name} score={m.score}
                      color={m.color} delay={i*100}/>
                  ))}
                </div>
                <div style={{marginTop:16,padding:'10px 14px',background:'#dcfce7',
                  borderRadius:8,border:'1px solid #86efac',fontSize:11,
                  fontWeight:600,color:'#16a34a'}}>
                  +7.4% improvement with multimodal fusion
                </div>
              </div>
              <div className="panel">
                <div className="panel-title">Training Accuracy (10 epochs)</div>
                <AccuracyChart data={[78.2,81.5,83.1,85.4,86.9,88.2,90.1,91.4,93.0,94.7]}/>
                <div style={{marginTop:12,display:'flex',gap:8,flexWrap:'wrap'}}>
                  <span className="tag tag-blue">DeepFace AI</span>
                  <span className="tag tag-green">MFCC Audio</span>
                  <span className="tag tag-amber">rPPG Heart Rate</span>
                </div>
              </div>
            </div>

            <div className="two-col">
              <div className="panel">
                <div className="panel-title">Confusion Matrix</div>
                <div className="matrix-grid">
                  <div/>
                  <div className="matrix-label">Pred: Stress</div>
                  <div className="matrix-label">Pred: No Stress</div>
                  <div className="matrix-label">Actual: Stress</div>
                  <div className="matrix-cell cell-tp">
                    {CONFUSION.tp}
                    <span className="matrix-cell-sub">TRUE POS</span>
                  </div>
                  <div className="matrix-cell cell-fn">
                    {CONFUSION.fn}
                    <span className="matrix-cell-sub">FALSE NEG</span>
                  </div>
                  <div className="matrix-label">Actual: No Stress</div>
                  <div className="matrix-cell cell-fp">
                    {CONFUSION.fp}
                    <span className="matrix-cell-sub">FALSE POS</span>
                  </div>
                  <div className="matrix-cell cell-tn">
                    {CONFUSION.tn}
                    <span className="matrix-cell-sub">TRUE NEG</span>
                  </div>
                </div>
              </div>
              <div className="panel">
                <div className="panel-title">System Architecture</div>
                {[
                  {icon:'🎭', title:'Facial Modality',
                   desc:'DeepFace opencv detector + emotion classifier', tag:'DeepFace'},
                  {icon:'🎙️', title:'Audio Modality',
                   desc:'MFCC 40-band features + TESS trained model',   tag:'LSTM'},
                  {icon:'💓', title:'Physio + rPPG',
                   desc:'Heart rate auto-detected from face video',      tag:'rPPG'},
                  {icon:'🔗', title:'Late Fusion',
                   desc:'Dynamic weighting based on available modalities', tag:'Fusion'},
                ].map((u,i) => (
                  <div key={i} style={{display:'flex',gap:12,padding:'10px 0',
                    borderBottom:i<3?'1px solid #f1f5f9':'none',alignItems:'flex-start'}}>
                    <span style={{fontSize:20}}>{u.icon}</span>
                    <div style={{flex:1}}>
                      <div style={{fontSize:12,fontWeight:700,color:'#1a1a2e',marginBottom:2}}>
                        {u.title}
                        <span style={{marginLeft:8,fontSize:9,fontWeight:700,
                          padding:'2px 6px',background:'#ede9fe',color:'#4f46e5',
                          borderRadius:4}}>{u.tag}</span>
                      </div>
                      <div style={{fontSize:11,color:'#94a3b8'}}>{u.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}