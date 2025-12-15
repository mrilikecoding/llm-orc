import { signal } from '@preact/signals'
import { useEffect } from 'preact/hooks'
import { api, Script, ScriptDetail } from '../api/client'

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
      className={`p-3 border rounded cursor-pointer transition-colors
        ${isSelected ? 'border-accent bg-accent/10' : 'border-border hover:border-text-secondary'}`}
      onClick={() => selectScript(script)}
    >
      <div className="font-medium text-text-primary">{script.name}</div>
      <div className="text-xs text-text-muted mt-1 font-mono truncate">{script.path}</div>
    </div>
  )
}

function ScriptDetailPanel() {
  const script = selectedScript.value
  if (!script) return null

  return (
    <div className="bg-bg-secondary border border-border rounded-lg overflow-hidden">
      <div className="p-4 border-b border-border flex justify-between items-center">
        <div>
          <h3 className="m-0 text-lg font-semibold">{script.name}</h3>
          <p className="m-0 mt-1 text-text-secondary text-sm">
            Category: <span className="text-accent">{script.category || 'uncategorized'}</span>
          </p>
        </div>
        <button
          className="text-text-secondary hover:text-text-primary text-xl"
          onClick={() => (selectedScript.value = null)}
        >
          ×
        </button>
      </div>

      {loadingDetail.value ? (
        <div className="p-6 text-text-secondary">Loading...</div>
      ) : (
        <div className="p-4">
          <div className="mb-4">
            <div className="text-sm font-semibold text-text-secondary mb-2">Path</div>
            <code className="block p-2 bg-bg-primary border border-border rounded text-sm">
              {script.path}
            </code>
          </div>

          {script.content && (
            <div className="mb-4">
              <div className="text-sm font-semibold text-text-secondary mb-2">Content</div>
              <pre className="p-3 bg-bg-primary border border-border rounded text-sm overflow-auto max-h-[200px]">
                {script.content}
              </pre>
            </div>
          )}

          <div className="border-t border-border pt-4 mt-4">
            <div className="text-sm font-semibold text-text-secondary mb-2">Test Script</div>
            <textarea
              className="w-full p-3 bg-bg-primary border border-border rounded text-text-primary
                font-mono text-sm resize-y focus:outline-none focus:border-accent"
              placeholder="Enter test input..."
              value={testInput.value}
              onInput={(e) => (testInput.value = (e.target as HTMLTextAreaElement).value)}
              rows={3}
            />
            <button
              onClick={runTest}
              disabled={testing.value || !testInput.value.trim()}
              className="mt-2 px-4 py-2 bg-accent hover:bg-accent/80 text-white rounded font-medium
                disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {testing.value ? 'Running...' : 'Run Test'}
            </button>

            {testOutput.value && (
              <div className="mt-4">
                <div className="text-sm font-semibold text-text-secondary mb-2">Output</div>
                <pre className={`p-3 border rounded text-sm overflow-auto max-h-[200px] whitespace-pre-wrap
                  ${testOutput.value.startsWith('Error:')
                    ? 'bg-error-bg/20 border-error text-error'
                    : 'bg-success-bg/20 border-success text-text-primary'}`}>
                  {testOutput.value}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export function ScriptsPage() {
  useEffect(() => {
    loadScripts()
  }, [])

  if (loading.value) {
    return <div>Loading scripts...</div>
  }

  if (error.value) {
    return <div className="text-error">Error: {error.value}</div>
  }

  const grouped = getGroupedScripts()
  const categories = Object.keys(grouped).sort()

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Scripts</h2>

      {scripts.value.length === 0 ? (
        <div className="text-center p-12 text-text-secondary">
          <p>No scripts found.</p>
          <p className="mt-2">Add scripts to .llm-orc/scripts/ to see them here.</p>
        </div>
      ) : (
        <div className="grid grid-cols-[300px_1fr] gap-6">
          <div className="space-y-4">
            {categories.map((category) => (
              <div key={category}>
                <div className="text-sm font-semibold text-text-secondary mb-2 uppercase tracking-wider">
                  {category}
                </div>
                <div className="space-y-2">
                  {grouped[category].map((script) => (
                    <ScriptCard key={`${script.category}/${script.name}`} script={script} />
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div>
            {selectedScript.value ? (
              <ScriptDetailPanel />
            ) : (
              <div className="text-center p-12 text-text-secondary border border-border border-dashed rounded-lg">
                Select a script to view details and test it
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
