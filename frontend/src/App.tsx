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
    const fetchHealth = async () => {
      try {
        const healthData = await checkHealth();
        setHealth(healthData);
      } catch (error) {
        console.error('Failed to fetch health status:', error);
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 10000); // Check every 10 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app">
      <Header />
      <StatusBar health={health} />
      <Chat />
    </div>
  );
}

export default App;
