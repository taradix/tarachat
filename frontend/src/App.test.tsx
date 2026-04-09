import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import App from './App'
import * as api from './api'

vi.mock('./api', () => ({
  checkHealth: vi.fn(),
  sendMessageStream: vi.fn(),
}))

describe('App', () => {
  beforeEach(() => {
    vi.resetAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('shows connecting state initially', () => {
    vi.mocked(api.checkHealth).mockReturnValue(new Promise(() => {}))

    render(<App />)
    expect(screen.getByText('Connecting to server...')).toBeInTheDocument()
  })

  it('shows ready when health check succeeds', async () => {
    vi.mocked(api.checkHealth).mockResolvedValue({
      status: 'healthy',
    })

    render(<App />)

    await waitFor(() => {
      expect(screen.getByText('Ready')).toBeInTheDocument()
    })
  })

  it('shows initializing when model not loaded', async () => {
    vi.mocked(api.checkHealth).mockResolvedValue({
      status: 'initializing',
    })

    render(<App />)

    await waitFor(() => {
      expect(screen.getByText('Initializing model...')).toBeInTheDocument()
    })
  })

  it('renders header and chat components', async () => {
    vi.mocked(api.checkHealth).mockResolvedValue({
      status: 'healthy',
    })

    render(<App />)

    await waitFor(() => {
      expect(screen.getByText('Taram Chat')).toBeInTheDocument()
    })

    expect(screen.getByText('Welcome to TaraChat!')).toBeInTheDocument()
  })

  it('keeps showing connecting on health check error', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    vi.mocked(api.checkHealth).mockRejectedValue(new Error('Network error'))

    render(<App />)

    // Wait for the effect to run
    await waitFor(() => {
      expect(api.checkHealth).toHaveBeenCalled()
    })

    expect(screen.getByText('Connecting to server...')).toBeInTheDocument()
  })
})
