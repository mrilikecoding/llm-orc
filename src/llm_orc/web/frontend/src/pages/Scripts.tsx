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
    <article
      className={`card${isSelected ? ' selected' : ''}`}
      onClick={() => selectScript(script)}
    >
      <header><strong>{script.name}</strong></header>
      <p><small><code>{script.path}</code></small></p>
    </article>
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
        <p aria-busy="true">Loading...</p>
      ) : script ? (
        <>
          <div style={{ marginBottom: '1rem' }}>
            <p className="muted-label">Path</p>
            <code>{script.path}</code>
          </div>

          {script.content && (
            <div style={{ marginBottom: '1rem' }}>
              <p className="muted-label">Content</p>
              <pre style={{ maxHeight: '200px', overflow: 'auto' }}><code>{script.content}</code></pre>
            </div>
          )}

          <hr />

          <p className="muted-label">Test Script</p>
          <label>
            <textarea
              placeholder="Enter test input..."
              value={testInput.value}
              onInput={(e) => (testInput.value = (e.target as HTMLTextAreaElement).value)}
              rows={3}
            />
          </label>
          <button
            onClick={runTest}
            disabled={testing.value || !testInput.value.trim()}
            aria-busy={testing.value}
          >
            {testing.value ? 'Running...' : 'Run Test'}
          </button>

          {testOutput.value && (
            <div style={{ marginTop: '1rem' }}>
              <p className="muted-label">Output</p>
              <pre style={{
                maxHeight: '200px',
                overflow: 'auto',
                borderColor: testOutput.value.startsWith('Error:') ? '#f85149' : '#3fb950',
              }}>
                <code>{testOutput.value}</code>
              </pre>
            </div>
          )}
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
    return <p aria-busy="true">Loading scripts...</p>
  }

  if (error.value) {
    return <p style={{ color: '#f85149' }}>Error: {error.value}</p>
  }

  const grouped = getGroupedScripts()
  const categories = Object.keys(grouped).sort()

  return (
    <div>
      <div className="page-header">
        <div>
          <h1>Scripts</h1>
          <p>{scripts.value.length} script{scripts.value.length !== 1 ? 's' : ''} available</p>
        </div>
      </div>

      {scripts.value.length === 0 ? (
        <div className="empty-state">
          <p>No scripts found.</p>
          <p><small>Add scripts to .llm-orc/scripts/</small></p>
        </div>
      ) : (
        <div className="spaced">
          {categories.map((category) => (
            <div key={category}>
              <p className="muted-label">{category}</p>
              <div className="card-grid">
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
