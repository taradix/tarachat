import { useState, useEffect } from 'react';
import Chat from './components/Chat';
import Header from './components/Header';
import StatusBar from './components/StatusBar';
import { checkHealth } from './api';
import { HealthResponse } from './types';
import './App.css';

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    const isHealthy = health?.status === 'healthy';
    const pollInterval = isHealthy ? 30000 : 10000;

    const fetchHealth = async () => {
      try {
        const healthData = await checkHealth();
        setHealth(healthData);
      } catch (error) {
        console.error('Failed to fetch health status:', error);
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, pollInterval);

    return () => clearInterval(interval);
  }, [health?.status]);

  return (
    <div className="app">
      <Header />
      <StatusBar health={health} />
      <Chat />
    </div>
  );
}

export default App;
