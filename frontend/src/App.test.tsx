import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import App from './App'

vi.mock('./api', () => ({
  sendMessageStream: vi.fn(),
}))

describe('App', () => {
  it('renders header and chat components', () => {
    render(<App />)
    expect(screen.getByText('Taram Chat')).toBeInTheDocument()
    expect(screen.getByText('Bienvenue sur TaraChat !')).toBeInTheDocument()
  })
})
