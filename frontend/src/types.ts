export interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
  timestamp: Date;
}

export interface ChatRequest {
  message: string;
  conversation_history?: Array<{
    role: string;
    content: string;
  }>;
}

export interface ChatResponse {
  response: string;
  sources: string[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  vector_store_ready: boolean;
}
