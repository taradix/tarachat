import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import MessageBubble from './MessageBubble'
import { Message } from '../types'

describe('MessageBubble', () => {
  const baseMessage: Message = {
    role: 'user',
    content: 'Hello world',
    timestamp: new Date('2026-01-15T10:30:00'),
  }

  it('renders user message content', () => {
    render(<MessageBubble message={baseMessage} />)
    expect(screen.getByText('Hello world')).toBeInTheDocument()
  })

  it('applies user class for user messages', () => {
    const { container } = render(<MessageBubble message={baseMessage} />)
    expect(container.querySelector('.user')).toBeInTheDocument()
  })

  it('applies assistant class for assistant messages', () => {
    const msg: Message = { ...baseMessage, role: 'assistant' }
    const { container } = render(<MessageBubble message={msg} />)
    expect(container.querySelector('.assistant')).toBeInTheDocument()
  })

  it('displays timestamp', () => {
    render(<MessageBubble message={baseMessage} />)
    expect(screen.getByText(/\d{2}:\d{2}/)).toBeInTheDocument()
  })

  it('shows sources when present', () => {
    const msg: Message = {
      ...baseMessage,
      role: 'assistant',
      sources: [
        { filename: 'doc1.pdf', page: 1, snippet: 'text1' },
        { filename: 'doc2.pdf', page: 3, snippet: 'text2' },
      ],
    }
    render(<MessageBubble message={msg} />)
    expect(screen.getByText('Sources (2)')).toBeInTheDocument()
    expect(screen.getByText('doc1.pdf')).toBeInTheDocument()
    expect(screen.getByText('doc2.pdf')).toBeInTheDocument()
  })

  it('hides sources section when sources is empty', () => {
    const msg: Message = { ...baseMessage, sources: [] }
    render(<MessageBubble message={msg} />)
    expect(screen.queryByText(/Sources/)).not.toBeInTheDocument()
  })

  it('hides sources section when sources is undefined', () => {
    render(<MessageBubble message={baseMessage} />)
    expect(screen.queryByText(/Sources/)).not.toBeInTheDocument()
  })
})
