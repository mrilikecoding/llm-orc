import { signal } from '@preact/signals'
import { useEffect } from 'preact/hooks'
import { api, Script, ScriptDetail } from '../api/client'
import { SlidePanel } from '../components/SlidePanel'

const scripts = signal<Script[]>([])
const loading = signal(true)
const error = signal<string | null>(null)
const selectedScript = signal<ScriptDetail | null>(null)
const loadingDetail = signal(false)
const testInput = signal('')
const testOutput = signal<string | null>(null)
const testing = signal(false)

async function loadScripts() {
  loading.value = true
  error.value = null
  try {
    const result = await api.scripts.list()
    scripts.value = result.scripts
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load scripts'
  } finally {
    loading.value = false
  }
}

async function selectScript(script: Script) {
  loadingDetail.value = true
  testOutput.value = null
  testInput.value = ''
  try {
    selectedScript.value = await api.scripts.get(script.category, script.name)
  } catch {
    selectedScript.value = { ...script }
  } finally {
    loadingDetail.value = false
  }
}

async function runTest() {
  if (!selectedScript.value || !testInput.value.trim()) return

  testing.value = true
  testOutput.value = null
  try {
    const result = await api.scripts.test(
      selectedScript.value.category,
      selectedScript.value.name,
      testInput.value
    )
    testOutput.value = result.success
      ? result.output || 'Success (no output)'
      : `Error: ${result.error}`
  } catch (e) {
    testOutput.value = `Error: ${e instanceof Error ? e.message : 'Unknown error'}`
  } finally {
    testing.value = false
  }
}

// Group scripts by category
function getGroupedScripts(): Record<string, Script[]> {
  const grouped: Record<string, Script[]> = {}
  for (const script of scripts.value) {
    const cat = script.category || 'uncategorized'
    if (!grouped[cat]) grouped[cat] = []
    grouped[cat].push(script)
  }
  return grouped
}

function ScriptCard({ script }: { script: Script }) {
  const isSelected =
    selectedScript.value?.name === script.name &&
    selectedScript.value?.category === script.category

  return (
    <div
      className={`bg-bg-secondary border rounded-xl p-6 cursor-pointer transition-all
        hover:shadow-xl hover:-translate-y-1
        ${isSelected ? 'border-accent ring-2 ring-accent/20' : 'border-border hover:border-accent/50'}`}
      onClick={() => selectScript(script)}
    >
      <h3 className="text-lg font-semibold text-text-primary mb-3">{script.name}</h3>
      <p className="text-xs text-text-muted font-mono truncate">{script.path}</p>
    </div>
  )
}

function ScriptDetailPanel() {
  const script = selectedScript.value

  return (
    <SlidePanel
      open={script !== null}
      onClose={() => (selectedScript.value = null)}
      title={script?.name || ''}
      subtitle={`Category: ${script?.category || 'uncategorized'}`}
      width="lg"
    >
      {loadingDetail.value ? (
        <div className="text-text-secondary py-8 text-center">Loading...</div>
      ) : script ? (
        <>
          {/* Path */}
          <div className="mb-4">
            <div className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
              Path
            </div>
            <code className="block p-3 bg-bg-primary border border-border rounded text-sm">
              {script.path}
            </code>
          </div>

          {/* Content preview */}
          {script.content && (
            <div className="mb-4">
              <div className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
                Content
              </div>
              <pre className="p-3 bg-bg-primary border border-border rounded text-xs overflow-auto max-h-[200px]">
                {script.content}
              </pre>
            </div>
          )}

          {/* Test runner */}
          <div className="border-t border-border pt-4 mt-4">
            <div className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
              Test Script
            </div>
            <textarea
              className="w-full p-3 bg-bg-primary border border-border rounded text-text-primary
                font-mono text-sm resize-y focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent/50"
              placeholder="Enter test input..."
              value={testInput.value}
              onInput={(e) => (testInput.value = (e.target as HTMLTextAreaElement).value)}
              rows={3}
            />
            <button
              onClick={runTest}
              disabled={testing.value || !testInput.value.trim()}
              className={`mt-3 w-full py-2.5 px-4 rounded-lg text-white font-medium transition-all
                ${testing.value || !testInput.value.trim()
                  ? 'bg-border-light cursor-not-allowed'
                  : 'gradient-button hover:-translate-y-0.5'}`}
            >
              {testing.value ? 'Running...' : 'Run Test'}
            </button>

            {testOutput.value && (
              <div className="mt-4">
                <div className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
                  Output
                </div>
                <pre className={`p-3 border rounded text-sm overflow-auto max-h-[200px] whitespace-pre-wrap
                  ${testOutput.value.startsWith('Error:')
                    ? 'bg-error/10 border-error/50 text-error'
                    : 'bg-success/10 border-success/50 text-text-primary'}`}>
                  {testOutput.value}
                </pre>
              </div>
            )}
          </div>
        </>
      ) : null}
    </SlidePanel>
  )
}

export function ScriptsPage() {
  useEffect(() => {
    loadScripts()
  }, [])

  if (loading.value) {
    return <div className="text-text-secondary">Loading scripts...</div>
  }

  if (error.value) {
    return <div className="text-error">Error: {error.value}</div>
  }

  const grouped = getGroupedScripts()
  const categories = Object.keys(grouped).sort()

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold">Scripts</h1>
          <p className="text-text-secondary text-sm mt-1">
            {scripts.value.length} script{scripts.value.length !== 1 ? 's' : ''} available
          </p>
        </div>
      </div>

      {scripts.value.length === 0 ? (
        <div className="text-center py-16 text-text-secondary">
          <div className="text-4xl mb-4 opacity-50">📜</div>
          <p>No scripts found.</p>
          <p className="text-sm mt-1">Add scripts to .llm-orc/scripts/</p>
        </div>
      ) : (
        <div className="space-y-8">
          {categories.map((category) => (
            <div key={category}>
              <div className="text-xs font-medium text-text-muted uppercase tracking-wider mb-3">
                {category}
              </div>
              <div className="grid grid-cols-[repeat(auto-fill,minmax(250px,1fr))] gap-5">
                {grouped[category].map((script) => (
                  <ScriptCard key={`${script.category}/${script.name}`} script={script} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      <ScriptDetailPanel />
    </div>
  )
}
