import { Message } from '../types';
import { SourceLink } from './PdfViewer';
import CitationText from './CitationText';
import './MessageBubble.css';

interface MessageBubbleProps {
  message: Message;
}

function MessageBubble({ message }: MessageBubbleProps) {
  return (
    <div className={`message-bubble ${message.role === 'user' ? 'user' : 'assistant'}`}>
      <div className="message-content">
        <div className="message-text">
          <CitationText content={message.content} sources={message.sources} />
        </div>
        {message.sources && message.sources.length > 0 && (
          <div className="message-sources">
            <details>
              <summary>Sources ({message.sources.length})</summary>
              <div className="sources-list">
                {message.sources.map((source, index) => (
                  <SourceLink key={index} source={source} />
                ))}
              </div>
            </details>
          </div>
        )}
      </div>
      <div className="message-timestamp">
        {message.timestamp.toLocaleTimeString('fr-CA', {
          hour: '2-digit',
          minute: '2-digit',
        })}
      </div>
    </div>
  );
}

export default MessageBubble;
