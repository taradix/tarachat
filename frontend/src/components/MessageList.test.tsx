import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import MessageList from './MessageList'
import { Message } from '../types'

describe('MessageList', () => {
  it('shows welcome message when empty', () => {
    render(<MessageList messages={[]} isLoading={false} />)
    expect(screen.getByText('Bienvenue sur TaraChat !')).toBeInTheDocument()
    expect(
      screen.getByText('Commencez une conversation en tapant un message ci-dessous.'),
    ).toBeInTheDocument()
  })

  it('renders messages when present', () => {
    const messages: Message[] = [
      {
        role: 'user',
        content: 'Hello',
        timestamp: new Date(),
      },
      {
        role: 'assistant',
        content: 'Hi there!',
        timestamp: new Date(),
      },
    ]
    render(<MessageList messages={messages} isLoading={false} />)
    expect(screen.getByText('Hello')).toBeInTheDocument()
    expect(screen.getByText('Hi there!')).toBeInTheDocument()
  })

  it('shows loading indicator when isLoading is true', () => {
    const messages: Message[] = [
      {
        role: 'user',
        content: 'Hello',
        timestamp: new Date(),
      },
    ]
    const { container } = render(
      <MessageList messages={messages} isLoading={true} />,
    )
    expect(container.querySelector('.loading-indicator')).toBeInTheDocument()
  })

  it('hides loading indicator when isLoading is false', () => {
    const messages: Message[] = [
      {
        role: 'user',
        content: 'Hello',
        timestamp: new Date(),
      },
    ]
    const { container } = render(
      <MessageList messages={messages} isLoading={false} />,
    )
    expect(
      container.querySelector('.loading-indicator'),
    ).not.toBeInTheDocument()
  })
})
