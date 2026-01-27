import { ComponentChildren } from 'preact'

interface SlidePanelProps {
  open: boolean
  onClose: () => void
  title: string
  subtitle?: string
  children: ComponentChildren
  width?: 'md' | 'lg' | 'xl'
}

export function SlidePanel({
  open,
  onClose,
  title,
  subtitle,
  children,
  width = 'lg',
}: SlidePanelProps) {
  if (!open) return null

  return (
    <>
      <div className="slide-backdrop" onClick={onClose} />
      <div className={`slide-panel width-${width}`}>
        <div className="slide-panel-header">
          <div>
            <h2>{title}</h2>
            {subtitle && <p>{subtitle}</p>}
          </div>
          <button
            onClick={onClose}
            className="outline secondary"
            style={{ padding: '0.25rem 0.5rem', margin: 0 }}
          >
            &times;
          </button>
        </div>
        <div className="slide-panel-body">
          {children}
        </div>
      </div>
    </>
  )
}
