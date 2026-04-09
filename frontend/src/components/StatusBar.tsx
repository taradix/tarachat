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

  const isReady = health.status === 'healthy';

  return (
    <div className={`status-bar ${isReady ? 'ready' : 'initializing'}`}>
      <span className="status-indicator"></span>
      {isReady ? 'Ready' : 'Initializing model...'}
    </div>
  );
}

export default StatusBar;
