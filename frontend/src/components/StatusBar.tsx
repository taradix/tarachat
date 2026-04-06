import { HealthResponse } from '../types';
import './StatusBar.css';

interface StatusBarProps {
  health: HealthResponse | null;
}

function StatusBar({ health }: StatusBarProps) {
  if (!health) {
    return (
      <div className="status-bar warning">
        <span className="status-indicator"></span>
        Connecting to server...
      </div>
    );
  }

  const isReady = health.status === 'healthy' && health.model_loaded;

  return (
    <div className={`status-bar ${isReady ? 'ready' : 'initializing'}`}>
      <span className="status-indicator"></span>
      {isReady ? 'Ready' : 'Initializing model...'}
      <span className="status-details">
        {health.model_loaded ? '✓ Model' : '⏳ Model'}
        {' | '}
        {health.vector_store_ready ? '✓ Vector Store' : '⏳ Vector Store'}
      </span>
    </div>
  );
}

export default StatusBar;
