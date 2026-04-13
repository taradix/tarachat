import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import Header from './Header'

describe('Header', () => {
  it('renders the title', () => {
    render(<Header />)
    expect(
      screen.getByText('Règlements - Municipalité de Notre-Dame-du-Laus'),
    ).toBeInTheDocument()
  })

  it('renders inside a header element', () => {
    render(<Header />)
    expect(screen.getByRole('banner')).toBeInTheDocument()
  })
})
