import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'

// ─── Constants ────────────────────────────────────────────────────────────────
const WS_BASE_URL = 'wss://shivanshjoshi-openenv-email-triage.hf.space'

const CATEGORY_COLORS = {
  Billing:   { bg: '#1e3a5f', border: '#3b82f6', text: '#60a5fa' },
  Technical: { bg: '#1a3a2a', border: '#22c55e', text: '#4ade80' },
  Sales:     { bg: '#2e1f5e', border: '#a855f7', text: '#c084fc' },
  Spam:      { bg: '#3a1a1a', border: '#ef4444', text: '#f87171' },
  Other:     { bg: '#2a2a2a', border: '#6b7280', text: '#9ca3af' },
}

const LOG_TYPE_STYLES = {
  info:     { color: '#64748b' },
  read:     { color: '#38bdf8' },
  action:   { color: '#a78bfa' },
  reward:   { color: '#34d399' },
  db:       { color: '#fb923c' },
  dbresult: { color: '#fbbf24' },
  draft:    { color: '#818cf8' },
  success:  { color: '#4ade80' },
  penalty:  { color: '#f87171' },
}

// ─── Sub-Components ───────────────────────────────────────────────────────────

function EmailCard({ email, isSelected, onClick }) {
  const catStyle = email.category ? CATEGORY_COLORS[email.category] : null

  return (
    <div
      id={`email-card-${email.id}`}
      className={`email-card ${isSelected ? 'selected' : ''}`}
      onClick={() => onClick(email)}
    >
      <div className="email-card-header">
        <span className="email-avatar">{email.sender[0].toUpperCase()}</span>
        <div className="email-meta">
          <span className="email-sender">{email.sender}</span>
          <span className="email-id">{email.id}</span>
        </div>
        {email.category && (
          <span
            className="category-badge"
            style={{
              background: catStyle.bg,
              border: `1px solid ${catStyle.border}`,
              color: catStyle.text,
            }}
          >
            {email.category}
          </span>
        )}
      </div>
      <div className="email-subject">{email.subject}</div>
      <div className="email-snippet">{email.snippet}</div>
    </div>
  )
}

function LogEntry({ entry, index }) {
  const style = LOG_TYPE_STYLES[entry.type] || LOG_TYPE_STYLES.info
  return (
    <div className={`log-entry log-${entry.type}`} style={{ animationDelay: `${index * 0.02}s` }}>
      <span className="log-time">{entry.time}</span>
      <span className="log-dot" style={{ background: style.color }} />
      <span
        className="log-text"
        style={{
          color: entry.type === 'success' ? '#4ade80' : entry.type === 'penalty' ? '#f87171' : undefined,
        }}
      >
        {entry.text}
      </span>
      {entry.reward !== undefined && entry.reward !== 0 && (
        <span
          className="log-reward"
          style={{ color: entry.reward > 0 ? '#34d399' : '#f87171' }}
        >
          {entry.reward > 0 ? `+${entry.reward.toFixed(2)}` : entry.reward.toFixed(2)}
        </span>
      )}
    </div>
  )
}

function RewardGauge({ score }) {
  const min = -1
  const max = 3
  const pct = Math.min(100, Math.max(0, ((score - min) / (max - min)) * 100))
  const color = score >= 0 ? '#34d399' : '#f87171'
  return (
    <div className="reward-gauge">
      <div className="reward-bar-track">
        <div className="reward-bar-fill" style={{ width: `${pct}%`, background: color }} />
        <div className="reward-bar-marker" style={{ left: `${((0 - min) / (max - min)) * 100}%` }} />
      </div>
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [activeTask, setActiveTask]               = useState(null)
  const [running, setRunning]                     = useState(false)
  const [emails, setEmails]                       = useState([])
  const [selectedEmail, setSelectedEmail]         = useState(null)
  const [logs, setLogs]                           = useState([])
  const [currentStep, setCurrentStep]             = useState(0)
  const [maxStep, setMaxStep]                     = useState(0)
  const [cumulativeReward, setCumulativeReward]   = useState(0)
  const [dbResult, setDbResult]                   = useState(null)
  const [taskDone, setTaskDone]                   = useState(false)
  const [finalScore, setFinalScore]               = useState(null)

  const logEndRef = useRef(null)
  const wsRef     = useRef(null)

  // Auto-scroll the log terminal
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  // Clean up WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [])

  const addLog = useCallback((event) => {
    const time = new Date().toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
    setLogs(prev => [...prev, { ...event, time, id: Date.now() + Math.random() }])
  }, [])

  const handleWsMessage = useCallback((msgEvent) => {
    let data
    try {
      data = JSON.parse(msgEvent.data)
    } catch {
      return
    }

    // Add to log feed
    addLog(data)

    // Update step and reward
    if (data.step !== undefined) setCurrentStep(data.step)
    if (data.max_steps !== undefined) setMaxStep(data.max_steps)
    if (data.cumulative_reward !== undefined) setCumulativeReward(data.cumulative_reward)

    // On init: populate inbox
    if (data.event === 'init' && data.inbox) {
      setEmails(data.inbox.map(e => ({ ...e, category: null })))
    }

    // On categorize: update badge
    if (data.category) {
      setEmails(prev =>
        prev.map(e =>
          e.id === data.category.id ? { ...e, category: data.category.cat } : e
        )
      )
    }

    // On DB result: show in panel
    if (data.db_result) {
      setDbResult(data.db_result)
    }

    // On completion
    if (data.done || data.event === 'complete') {
      setRunning(false)
      setTaskDone(true)
      if (data.final_score !== undefined) {
        setFinalScore(data.final_score)
      }
    }

    // On error
    if (data.event === 'error') {
      setRunning(false)
    }
  }, [addLog])

  const startTask = useCallback((taskId) => {
    if (running) return

    // Reset state
    setActiveTask(taskId)
    setRunning(true)
    setLogs([])
    setCurrentStep(0)
    setMaxStep(0)
    setCumulativeReward(0)
    setDbResult(null)
    setTaskDone(false)
    setFinalScore(null)
    setSelectedEmail(null)
    setEmails([])

    // Close existing WS
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    // Open new WebSocket
    const ws = new WebSocket(`${WS_BASE_URL}/ws/task/${taskId}`)
    wsRef.current = ws

    ws.onopen = () => {
      console.log(`[WS] Connected to task ${taskId}`)
    }

    ws.onmessage = handleWsMessage

    ws.onerror = (err) => {
      console.error('[WS] Error:', err)
      addLog({
        type: 'penalty',
        text: '❌ WebSocket connection error. Check if backend is running.',
      })
      setRunning(false)
    }

    ws.onclose = () => {
      console.log('[WS] Connection closed')
    }
  }, [running, handleWsMessage, addLog])

  const stopTask = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setRunning(false)
    addLog({ type: 'penalty', text: '⛔ Task manually stopped by user.' })
  }, [addLog])

  const resetDashboard = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setActiveTask(null)
    setRunning(false)
    setEmails([])
    setLogs([])
    setTaskDone(false)
    setFinalScore(null)
    setCumulativeReward(0)
    setCurrentStep(0)
    setMaxStep(0)
    setDbResult(null)
    setSelectedEmail(null)
  }, [])

  const progressPct = maxStep > 0 ? (currentStep / maxStep) * 100 : 0

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="app">
      {/* ── Top Nav ── */}
      <header className="topbar">
        <div className="topbar-brand">
          <span className="topbar-icon">📧</span>
          <div>
            <h1 className="topbar-title">OpenEnv Email Triage</h1>
            <p className="topbar-sub">AI Agent Monitoring Dashboard</p>
          </div>
        </div>
        <div className="topbar-center">
          <div className={`status-pill ${running ? 'running' : taskDone ? 'done' : 'idle'}`}>
            <span className="status-dot" />
            {running ? 'Agent Running' : taskDone ? 'Task Complete' : 'Awaiting Task'}
          </div>
        </div>
        <div className="topbar-right">
          {activeTask && <span className="active-task-label">Task {activeTask} Active</span>}
          <span className="topbar-version">v1.0.0</span>
        </div>
      </header>

      {/* ── Main Layout ── */}
      <div className="dashboard">

        {/* ═══ LEFT: Inbox Panel ═══ */}
        <aside className="panel panel-left" id="inbox-panel">
          <div className="panel-header">
            <span className="panel-icon">📥</span>
            <h2 className="panel-title">Inbox</h2>
            <span className="email-count-badge">{emails.length}</span>
          </div>
          <div className="email-list">
            {emails.length === 0 ? (
              <div className="empty-state">
                <p>🚀 Start a task to load emails</p>
              </div>
            ) : (
              emails.map(email => (
                <EmailCard
                  key={email.id}
                  email={email}
                  isSelected={selectedEmail?.id === email.id}
                  onClick={setSelectedEmail}
                />
              ))
            )}
          </div>
          {selectedEmail && (
            <div className="email-detail-pane">
              <div className="email-detail-header">
                <button className="close-btn" onClick={() => setSelectedEmail(null)}>✕</button>
                <strong>{selectedEmail.subject}</strong>
              </div>
              <p className="email-detail-from">From: {selectedEmail.sender}</p>
              <p className="email-detail-body">{selectedEmail.snippet}</p>
            </div>
          )}
        </aside>

        {/* ═══ CENTER: Agent Feed ═══ */}
        <main className="panel panel-center" id="agent-feed-panel">
          <div className="panel-header">
            <span className="panel-icon">🤖</span>
            <h2 className="panel-title">Live Agent Feed</h2>
            {running && (
              <span className="live-indicator">
                <span className="blink-dot" />
                LIVE
              </span>
            )}
          </div>

          {/* Task Buttons */}
          <div className="task-controls">
            {[1, 2, 3].map(id => (
              <button
                key={id}
                id={`start-task-${id}-btn`}
                className={`task-btn ${activeTask === id ? 'active' : ''} ${running && activeTask !== id ? 'disabled' : ''}`}
                onClick={() => startTask(id)}
                disabled={running}
              >
                <span className="task-btn-label">Task {id}</span>
                <span className="task-btn-diff">{{ 1: 'Easy', 2: 'Medium', 3: 'Hard' }[id]}</span>
              </button>
            ))}
            {running && (
              <button id="stop-task-btn" className="stop-btn" onClick={stopTask}>
                ⛔ Stop
              </button>
            )}
          </div>

          {/* Progress Bar */}
          {maxStep > 0 && (
            <div className="progress-bar-container">
              <div className="progress-bar-labels">
                <span>Step {currentStep} / {maxStep}</span>
                <span>{Math.round(progressPct)}% complete</span>
              </div>
              <div className="progress-bar-track">
                <div className="progress-bar-fill" style={{ width: `${progressPct}%` }} />
              </div>
            </div>
          )}

          {/* Log Terminal */}
          <div className="log-terminal" id="log-terminal">
            {logs.length === 0 ? (
              <div className="terminal-empty">
                <p className="terminal-prompt">{'>'} Waiting for agent instructions...</p>
                <p className="terminal-hint">Select a task above to begin. Backend must be running on port 8000.</p>
              </div>
            ) : (
              logs.map((entry, i) => <LogEntry key={entry.id} entry={entry} index={i} />)
            )}
            <div ref={logEndRef} />
          </div>
        </main>

        {/* ═══ RIGHT: Metrics Panel ═══ */}
        <aside className="panel panel-right" id="metrics-panel">
          <div className="panel-header">
            <span className="panel-icon">📊</span>
            <h2 className="panel-title">Metrics & State</h2>
          </div>

          {/* Reward Score */}
          <div className="metric-card reward-card">
            <p className="metric-label">Cumulative Reward</p>
            <p className={`reward-score ${cumulativeReward > 0 ? 'positive' : cumulativeReward < 0 ? 'negative' : ''}`}>
              {cumulativeReward >= 0 ? '+' : ''}{cumulativeReward.toFixed(2)}
            </p>
            <RewardGauge score={cumulativeReward} />
          </div>

          {/* Step Counter */}
          <div className="metric-card step-card">
            <p className="metric-label">Current Step</p>
            <div className="step-display">
              <span className="step-current">{currentStep}</span>
              <span className="step-sep">/</span>
              <span className="step-max">{maxStep || '—'}</span>
            </div>
            <p className="metric-label" style={{ marginTop: 8 }}>Active Task</p>
            <div className="active-task-display">
              {activeTask ? (
                <span className={`task-tag task-tag-${activeTask}`}>
                  Task {activeTask} — {{ 1: 'Easy', 2: 'Medium', 3: 'Hard' }[activeTask]}
                </span>
              ) : (
                <span className="no-task">None</span>
              )}
            </div>
          </div>

          {/* Statistics grid */}
          <div className="metric-card stats-grid-card">
            <p className="metric-label">Session Statistics</p>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-val">{emails.filter(e => e.category).length}</span>
                <span className="stat-key">Categorized</span>
              </div>
              <div className="stat-item">
                <span className="stat-val">{emails.filter(e => e.category === 'Spam').length}</span>
                <span className="stat-key">Spam</span>
              </div>
              <div className="stat-item">
                <span className="stat-val">{emails.filter(e => e.category === 'Billing').length}</span>
                <span className="stat-key">Billing</span>
              </div>
              <div className="stat-item">
                <span className="stat-val">{emails.length}</span>
                <span className="stat-key">Total Emails</span>
              </div>
            </div>
          </div>

          {/* DB Lookup Result */}
          <div className="metric-card db-card">
            <div className="db-card-header">
              <span className="panel-icon" style={{ fontSize: 14 }}>🗄️</span>
              <p className="metric-label" style={{ margin: 0 }}>Database Lookup</p>
              {dbResult && <span className="db-hit-badge">HIT</span>}
            </div>
            {dbResult ? (
              <div className="db-result">
                <pre className="db-json">{JSON.stringify(dbResult, null, 2)}</pre>
              </div>
            ) : (
              <div className="db-empty">
                <p>— No query yet —</p>
              </div>
            )}
          </div>

          {/* Final Score banner */}
          {taskDone && (
            <div className="final-score-card">
              <p className="final-score-label">🏆 Final Score</p>
              <p className="final-score-value">
                {finalScore !== null ? finalScore.toFixed(4) : cumulativeReward.toFixed(4)}
              </p>
              <button id="reset-btn" className="reset-btn" onClick={resetDashboard}>
                ↺ Reset Dashboard
              </button>
            </div>
          )}
        </aside>
      </div>
    </div>
  )
}
