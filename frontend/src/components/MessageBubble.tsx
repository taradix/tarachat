import { Message } from '../types';
import './MessageBubble.css';

interface MessageBubbleProps {
  message: Message;
}

function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`message-bubble ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-content">
        <div className="message-text">{message.content}</div>
        {message.sources && message.sources.length > 0 && (
          <div className="message-sources">
            <details>
              <summary>Sources ({message.sources.length})</summary>
              <div className="sources-list">
                {message.sources.map((source, index) => (
                  <div key={index} className="source-item">
                    {source}
                  </div>
                ))}
              </div>
            </details>
          </div>
        )}
      </div>
      <div className="message-timestamp">
        {message.timestamp.toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
        })}
      </div>
    </div>
  );
}

export default MessageBubble;
