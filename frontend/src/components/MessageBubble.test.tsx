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

  it('renders inline citation buttons for assistant messages with citation markers', () => {
    const msg: Message = {
      ...baseMessage,
      role: 'assistant',
      content: 'The answer is here [doc.pdf#page=5] and more [other.pdf#page=2].',
      sources: [
        { filename: 'doc.pdf', page: 5, snippet: 'answer' },
        { filename: 'other.pdf', page: 2, snippet: 'more' },
      ],
    }
    const { container } = render(<MessageBubble message={msg} />)
    const citations = container.querySelectorAll('.inline-citation')
    expect(citations).toHaveLength(2)
    expect(citations[0].textContent).toBe('1')
    expect(citations[1].textContent).toBe('2')
  })

  it('assigns the same number to duplicate citation references', () => {
    const msg: Message = {
      ...baseMessage,
      role: 'assistant',
      content: 'First [a.pdf#page=1] then [b.pdf#page=2] and again [a.pdf#page=1].',
      sources: [
        { filename: 'a.pdf', page: 1, snippet: 'first' },
        { filename: 'b.pdf', page: 2, snippet: 'then' },
      ],
    }
    const { container } = render(<MessageBubble message={msg} />)
    const citations = container.querySelectorAll('.inline-citation')
    expect(citations).toHaveLength(3)
    expect(citations[0].textContent).toBe('1')
    expect(citations[1].textContent).toBe('2')
    expect(citations[2].textContent).toBe('1')
  })
})
