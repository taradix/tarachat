import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import Chat from './Chat'
import * as api from '../api'

vi.mock('../api', () => ({
  sendMessageStream: vi.fn(),
}))

describe('Chat', () => {
  beforeEach(() => {
    vi.resetAllMocks()
  })

  it('shows empty state initially', () => {
    render(<Chat />)
    expect(screen.getByText('Welcome to TaraChat!')).toBeInTheDocument()
  })

  it('sends a message and displays streaming response', async () => {
    const user = userEvent.setup({ delay: null })

    vi.mocked(api.sendMessageStream).mockImplementation(
      async (_req, callbacks) => {
        callbacks.onToken('Hello ')
        callbacks.onToken('back!')
        callbacks.onSources([{ filename: 'doc.pdf', page: 1, snippet: 'text' }])
        callbacks.onDone()
      },
    )

    render(<Chat />)
    await user.type(
      screen.getByPlaceholderText(/Type your message/),
      'Hi{Enter}',
    )

    await waitFor(() => {
      expect(screen.getByText('Hi')).toBeInTheDocument()
      expect(screen.getByText('Hello back!')).toBeInTheDocument()
      expect(screen.getByText('Sources (1)')).toBeInTheDocument()
    })
  })

  it('shows error message when stream fails', async () => {
    const user = userEvent.setup({ delay: null })
    vi.spyOn(console, 'error').mockImplementation(() => {})

    vi.mocked(api.sendMessageStream).mockImplementation(
      async (_req, callbacks) => {
        callbacks.onError('Something went wrong')
        callbacks.onDone()
      },
    )

    render(<Chat />)
    await user.type(
      screen.getByPlaceholderText(/Type your message/),
      'Hi{Enter}',
    )

    await waitFor(() => {
      expect(
        screen.getByText(/Sorry, I encountered an error/),
      ).toBeInTheDocument()
    })
  })

  it('shows error message when sendMessageStream throws', async () => {
    const user = userEvent.setup({ delay: null })
    vi.spyOn(console, 'error').mockImplementation(() => {})

    vi.mocked(api.sendMessageStream).mockRejectedValue(
      new Error('Network error'),
    )

    render(<Chat />)
    await user.type(
      screen.getByPlaceholderText(/Type your message/),
      'Hi{Enter}',
    )

    await waitFor(() => {
      expect(
        screen.getByText(/Sorry, I encountered an error/),
      ).toBeInTheDocument()
    })
  })

  it('passes conversation history to sendMessageStream', async () => {
    const user = userEvent.setup({ delay: null })

    vi.mocked(api.sendMessageStream).mockImplementation(
      async (_req, callbacks) => {
        callbacks.onToken('Response')
        callbacks.onDone()
      },
    )

    render(<Chat />)

    await user.type(
      screen.getByPlaceholderText(/Type your message/),
      'First{Enter}',
    )

    await waitFor(() => {
      expect(screen.getByText('Response')).toBeInTheDocument()
    })

    await user.type(
      screen.getByPlaceholderText(/Type your message/),
      'Second{Enter}',
    )

    await waitFor(() => {
      expect(api.sendMessageStream).toHaveBeenCalledTimes(2)
    })

    const secondCall = vi.mocked(api.sendMessageStream).mock.calls[1]
    expect(secondCall[0].conversation_history).toEqual([
      { role: 'user', content: 'First' },
      { role: 'assistant', content: 'Response' },
    ])
  })
})
