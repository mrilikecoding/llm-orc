import { signal } from '@preact/signals'
import { useEffect } from 'preact/hooks'
import { api, Ensemble, EnsembleDetail as EnsembleDetailType, ExecutionResult } from '../api/client'

const ensembles = signal<Ensemble[]>([])
const loading = signal(true)
const error = signal<string | null>(null)
const selectedEnsemble = signal<EnsembleDetailType | null>(null)
const loadingDetail = signal(false)
const executeInput = signal('')
const executeResult = signal<ExecutionResult | null>(null)
const executing = signal(false)
const activeTab = signal<'agents' | 'execute' | 'config'>('agents')

async function loadEnsembles() {
  loading.value = true
  error.value = null
  try {
    ensembles.value = await api.ensembles.list()
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load ensembles'
  } finally {
    loading.value = false
  }
}

async function selectEnsemble(ensemble: Ensemble) {
  loadingDetail.value = true
  executeResult.value = null
  activeTab.value = 'agents'
  try {
    selectedEnsemble.value = await api.ensembles.get(ensemble.name)
  } catch {
    selectedEnsemble.value = { ...ensemble, agents: ensemble.agents || [] }
  } finally {
    loadingDetail.value = false
  }
}

async function executeEnsemble() {
  if (!selectedEnsemble.value || !executeInput.value.trim()) return
  executing.value = true
  executeResult.value = null
  try {
    executeResult.value = await api.ensembles.execute(
      selectedEnsemble.value.name,
      executeInput.value
    )
  } catch (e) {
    executeResult.value = {
      status: 'error',
      results: {},
      synthesis: e instanceof Error ? e.message : 'Unknown error',
    }
  } finally {
    executing.value = false
  }
}

function EnsembleCard({ ensemble }: { ensemble: Ensemble }) {
  const isSelected = selectedEnsemble.value?.name === ensemble.name
  const agentCount = ensemble.agents?.length || 0

  return (
    <div
      className={`bg-bg-secondary border rounded-lg p-4 cursor-pointer transition-colors
        ${isSelected ? 'border-accent' : 'border-border hover:border-text-secondary'}`}
      onClick={() => selectEnsemble(ensemble)}
    >
      <div className="text-lg font-semibold text-accent mb-2">{ensemble.name}</div>
      <div className="text-text-secondary text-sm">{ensemble.description}</div>
      <div className="mt-3 flex gap-3 text-xs text-text-muted">
        <span className="py-0.5 px-2 bg-border-light rounded-xl">{ensemble.source}</span>
        {agentCount > 0 && (
          <span className="py-0.5 px-2 bg-border-light rounded-xl">
            {agentCount} agent{agentCount !== 1 ? 's' : ''}
          </span>
        )}
      </div>
    </div>
  )
}

function AgentsTab() {
  const ens = selectedEnsemble.value
  if (!ens?.agents?.length) {
    return <div className="text-text-secondary">No agents configured</div>
  }

  return (
    <div className="flex flex-col gap-3">
      {ens.agents.map((agent) => (
        <div key={agent.name} className="bg-bg-primary border border-border rounded-md p-4">
          <div className="flex justify-between items-center mb-2">
            <span className="font-semibold text-text-primary">{agent.name}</span>
            <span className="text-sm text-text-secondary bg-border-light py-1 px-2 rounded">
              {agent.model_profile}
            </span>
          </div>
          {agent.role && <div className="text-sm text-text-secondary mb-2">{agent.role}</div>}
          {agent.script && (
            <div className="text-sm text-text-secondary mb-2">
              Script: <code className="text-success">{agent.script}</code>
            </div>
          )}
          {agent.depends_on && agent.depends_on.length > 0 && (
            <div className="text-sm text-text-muted flex gap-2 items-center flex-wrap">
              <span>Depends on:</span>
              {agent.depends_on.map((dep) => (
                <span key={dep} className="bg-accent/10 text-accent py-0.5 px-2 rounded text-xs">
                  {dep}
                </span>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

function ExecuteTab() {
  return (
    <div>
      <textarea
        className="w-full p-3 bg-bg-primary border border-border rounded-md text-text-primary
          font-sans text-sm resize-y focus:outline-none focus:border-accent"
        placeholder="Enter input for the ensemble..."
        value={executeInput.value}
        onInput={(e) => (executeInput.value = (e.target as HTMLTextAreaElement).value)}
        rows={4}
      />
      <button
        className={`py-2 px-4 rounded-md text-white font-medium mt-3 transition-colors
          ${executing.value ? 'bg-border-light cursor-not-allowed' : 'bg-success-bg hover:bg-success-bg/80 cursor-pointer'}`}
        onClick={executeEnsemble}
        disabled={executing.value}
      >
        {executing.value ? 'Executing...' : 'Execute Ensemble'}
      </button>

      {executeResult.value && (
        <div className="mt-4 bg-bg-primary rounded-md overflow-hidden">
          <div className="py-3 px-4 bg-bg-secondary border-b border-border flex justify-between items-center">
            <span className={`font-medium ${
              executeResult.value.status === 'success' ? 'text-success' : 'text-error'
            }`}>
              {executeResult.value.status === 'success' ? 'Completed' : 'Failed'}
            </span>
          </div>
          <div className="p-4 font-mono text-sm whitespace-pre-wrap overflow-auto max-h-[400px]">
            {Object.entries(executeResult.value.results).map(([name, result]) => (
              <div key={name} className="mb-4 pb-4 border-b border-border-light last:border-0">
                <div className="font-semibold text-accent mb-2">{name}</div>
                <div>{result.response || result.error || 'No output'}</div>
              </div>
            ))}
            {executeResult.value.synthesis && (
              <div className="bg-accent/5 border border-accent/25 rounded-md p-4 mt-4">
                <div className="text-sm text-accent mb-2 font-medium">Synthesis</div>
                <div>{executeResult.value.synthesis}</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function ConfigTab() {
  const ens = selectedEnsemble.value
  if (!ens) return null

  return (
    <div>
      <pre className="font-mono text-sm whitespace-pre-wrap overflow-auto max-h-[400px]
        bg-bg-primary border border-border rounded-md p-4">
        {JSON.stringify(ens, null, 2)}
      </pre>
    </div>
  )
}

function EnsembleDetailPanel() {
  const ens = selectedEnsemble.value
  if (!ens) return null

  return (
    <div className="bg-bg-secondary border border-border rounded-lg mt-6 overflow-hidden">
      <div className="p-4 border-b border-border flex justify-between items-center">
        <div>
          <h3 className="m-0 mb-1">{ens.name}</h3>
          <p className="m-0 text-text-secondary text-sm">{ens.description}</p>
        </div>
        <button
          className="bg-transparent border-none text-text-secondary cursor-pointer text-xl p-1
            hover:text-text-primary"
          onClick={() => (selectedEnsemble.value = null)}
        >
          ×
        </button>
      </div>

      <div className="flex gap-2 border-b border-border px-4">
        {(['agents', 'execute', 'config'] as const).map((tab) => (
          <button
            key={tab}
            className={`py-3 px-4 bg-transparent border-none cursor-pointer -mb-px
              border-b-2 transition-colors
              ${activeTab.value === tab
                ? 'text-text-primary border-accent'
                : 'text-text-secondary border-transparent hover:text-text-primary'}`}
            onClick={() => (activeTab.value = tab)}
          >
            {tab === 'agents' ? `Agents (${ens.agents?.length || 0})` : tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      <div className="p-6">
        {loadingDetail.value ? (
          <div className="text-text-secondary">Loading...</div>
        ) : (
          <>
            {activeTab.value === 'agents' && <AgentsTab />}
            {activeTab.value === 'execute' && <ExecuteTab />}
            {activeTab.value === 'config' && <ConfigTab />}
          </>
        )}
      </div>
    </div>
  )
}

export function EnsemblesPage() {
  useEffect(() => {
    loadEnsembles()
  }, [])

  if (loading.value) {
    return <div>Loading ensembles...</div>
  }

  if (error.value) {
    return <div className="text-error">Error: {error.value}</div>
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Ensembles</h2>
        <span className="text-text-secondary">{ensembles.value.length} available</span>
      </div>

      <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-4">
        {ensembles.value.map((e) => (
          <EnsembleCard key={e.name} ensemble={e} />
        ))}
      </div>

      <EnsembleDetailPanel />
    </div>
  )
}
