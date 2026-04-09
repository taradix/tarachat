import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import StatusBar from './StatusBar'
import { HealthResponse } from '../types'

describe('StatusBar', () => {
  it('shows connecting message when health is null', () => {
    render(<StatusBar health={null} />)
    expect(screen.getByText('Connecting to server...')).toBeInTheDocument()
  })

  it('applies warning class when health is null', () => {
    const { container } = render(<StatusBar health={null} />)
    expect(container.querySelector('.warning')).toBeInTheDocument()
  })

  it('shows ready when healthy', () => {
    const health: HealthResponse = { status: 'healthy' }
    render(<StatusBar health={health} />)
    expect(screen.getByText('Ready')).toBeInTheDocument()
  })

  it('applies ready class when healthy', () => {
    const health: HealthResponse = { status: 'healthy' }
    const { container } = render(<StatusBar health={health} />)
    expect(container.querySelector('.ready')).toBeInTheDocument()
  })

  it('shows initializing when not healthy', () => {
    const health: HealthResponse = { status: 'initializing' }
    render(<StatusBar health={health} />)
    expect(screen.getByText('Initializing model...')).toBeInTheDocument()
  })

  it('applies initializing class when not healthy', () => {
    const health: HealthResponse = { status: 'initializing' }
    const { container } = render(<StatusBar health={health} />)
    expect(container.querySelector('.initializing')).toBeInTheDocument()
  })
})
