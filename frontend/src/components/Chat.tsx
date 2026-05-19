import { useState, useRef, useEffect } from 'react';
import { Message } from '../types';
import { sendMessageStream } from '../api';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import './Chat.css';

const ERROR_MESSAGE = "Désolé, une erreur s'est produite. Veuillez vous assurer que le serveur est en marche et réessayer.";

const SUGGESTED_QUESTIONS = [
  'À quelle distance puis-je construire de la rive?',
  'Quelles sont les règles concernant les clôtures et haies?',
  'Quelles sont les heures permises pour les travaux bruyants?',
];

function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    // Add a placeholder assistant message for streaming
    const assistantMessage: Message = {
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, assistantMessage]);

    try {
      const history = messages.map((msg) => ({
        role: msg.role,
        content: msg.content,
      }));

      await sendMessageStream(
        { message: content, conversation_history: history },
        {
          onToken: (token) => {
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last.role === 'assistant') {
                updated[updated.length - 1] = { ...last, content: last.content + token };
              }
              return updated;
            });
          },
          onSources: (sources) => {
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last.role === 'assistant') {
                updated[updated.length - 1] = { ...last, sources };
              }
              return updated;
            });
          },
          onError: (error) => {
            console.error('Stream error:', error);
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last.role === 'assistant' && last.content === '') {
                updated[updated.length - 1] = {
                  ...last,
                  content: ERROR_MESSAGE,
                };
              }
              return updated;
            });
          },
          onDone: () => {
            setIsLoading(false);
          },
        },
      );
    } catch (error) {
      console.error('Failed to send message:', error);
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last.role === 'assistant') {
          updated[updated.length - 1] = {
            ...last,
            content: ERROR_MESSAGE,
          };
        }
        return updated;
      });
      setIsLoading(false);
    }
  };

  return (
    <div className="chat">
      <div className="chat-container">
        <MessageList messages={messages} isLoading={isLoading} />
        {messages.length === 0 && !isLoading && (
          <div className="suggested-questions">
            {SUGGESTED_QUESTIONS.map((question) => (
              <button
                key={question}
                className="suggested-question"
                onClick={() => handleSendMessage(question)}
              >
                {question}
              </button>
            ))}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <MessageInput onSendMessage={handleSendMessage} disabled={isLoading} />
    </div>
  );
}

export default Chat;
