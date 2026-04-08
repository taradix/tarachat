import axios from 'axios';
import { ChatRequest, HealthResponse } from './types';

const API_URL = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const checkHealth = async (): Promise<HealthResponse> => {
  const response = await api.get<HealthResponse>('/health');
  return response.data;
};

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onSources: (sources: string[]) => void;
  onError: (error: string) => void;
  onDone: () => void;
}

export const sendMessageStream = async (
  request: ChatRequest,
  callbacks: StreamCallbacks,
): Promise<void> => {
  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorText = await response.text();
    callbacks.onError(errorText || 'Request failed');
    callbacks.onDone();
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    callbacks.onError('No response stream available');
    callbacks.onDone();
    return;
  }

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      // Keep the last potentially incomplete line in the buffer
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data: ')) continue;

        const data = trimmed.slice(6);
        if (data === '[DONE]') {
          callbacks.onDone();
          return;
        }

        try {
          const parsed = JSON.parse(data);
          if (parsed.type === 'token') {
            callbacks.onToken(parsed.content);
          } else if (parsed.type === 'sources') {
            callbacks.onSources(parsed.sources);
          } else if (parsed.type === 'error') {
            callbacks.onError(parsed.content);
          }
        } catch {
          // Skip malformed JSON lines
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  callbacks.onDone();
};
