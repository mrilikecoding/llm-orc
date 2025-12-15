import { ComponentChildren } from 'preact'

interface SlidePanelProps {
  open: boolean
  onClose: () => void
  title: string
  subtitle?: string
  children: ComponentChildren
  width?: 'md' | 'lg' | 'xl'
}

const widthClasses = {
  md: 'w-[400px]',
  lg: 'w-[500px]',
  xl: 'w-[600px]',
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
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/40 z-40 transition-opacity"
        onClick={onClose}
      />

      {/* Panel */}
      <div
        className={`fixed top-0 right-0 h-full ${widthClasses[width]} bg-bg-secondary
          border-l border-border shadow-2xl z-50 flex flex-col
          animate-slide-in`}
      >
        {/* Header */}
        <div className="flex-shrink-0 p-4 border-b border-border flex justify-between items-start">
          <div>
            <h2 className="text-lg font-semibold m-0">{title}</h2>
            {subtitle && (
              <p className="text-sm text-text-secondary m-0 mt-1">{subtitle}</p>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-text-secondary hover:text-text-primary text-2xl leading-none
              w-8 h-8 flex items-center justify-center rounded hover:bg-border-light"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {children}
        </div>
      </div>
    </>
  )
}
