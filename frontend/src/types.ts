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
