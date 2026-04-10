export interface Source {
  filename: string;
  page: number;
  highlights: string[];
}

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  timestamp: Date;
}

export interface ChatRequest {
  message: string;
  conversation_history?: Array<{
    role: string;
    content: string;
  }>;
}
