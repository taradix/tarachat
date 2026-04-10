import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import Header from './Header'

describe('Header', () => {
  it('renders the title', () => {
    render(<Header />)
    expect(screen.getByText('Taram Chat')).toBeInTheDocument()
  })

  it('renders the subtitle', () => {
    render(<Header />)
    expect(
      screen.getByText('Bylaws - Municipality of Notre-Dame-du-Laus'),
    ).toBeInTheDocument()
  })

  it('renders inside a header element', () => {
    render(<Header />)
    expect(screen.getByRole('banner')).toBeInTheDocument()
  })
})
