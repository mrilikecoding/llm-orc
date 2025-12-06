const BASE_URL = '/api'

export interface Agent {
  name: string
  model_profile: string
  role?: string
  depends_on?: string[]
  script?: string
}

export interface Ensemble {
  name: string
  description: string
  source: string
  agents?: Agent[]
  synthesis_prompt?: string
}

export interface EnsembleDetail extends Ensemble {
  agents: Agent[]
  file_path?: string
}

export interface Profile {
  name: string
  provider: string
  model: string
}

export interface Artifact {
  name: string
  executions_count: number
  latest_execution: string
}

export interface ArtifactDetail {
  ensemble_name: string
  timestamp: string
  status: string
  total_duration_ms: number
  agents: {
    name: string
    status: string
    result?: string
    error?: string
    duration_ms?: number
  }[]
  synthesis?: string
}

export interface ExecutionResult {
  status: string
  results: Record<string, { response?: string; error?: string }>
  synthesis?: string
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  })
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`)
  }
  return response.json()
}

export const api = {
  ensembles: {
    list: () => fetchJson<Ensemble[]>(`${BASE_URL}/ensembles`),
    get: (name: string) => fetchJson<EnsembleDetail>(`${BASE_URL}/ensembles/${name}`),
    execute: (name: string, input: string) =>
      fetchJson<ExecutionResult>(
        `${BASE_URL}/ensembles/${name}/execute`,
        { method: 'POST', body: JSON.stringify({ input }) }
      ),
    validate: (name: string) =>
      fetchJson<{ valid: boolean; details: { errors: string[] } }>(
        `${BASE_URL}/ensembles/${name}/validate`,
        { method: 'POST' }
      ),
  },
  profiles: {
    list: () => fetchJson<Profile[]>(`${BASE_URL}/profiles`),
  },
  artifacts: {
    list: () => fetchJson<Artifact[]>(`${BASE_URL}/artifacts`),
    getForEnsemble: (ensemble: string) =>
      fetchJson<ArtifactDetail[]>(`${BASE_URL}/artifacts/${ensemble}`),
    get: (ensemble: string, id: string) =>
      fetchJson<ArtifactDetail>(`${BASE_URL}/artifacts/${ensemble}/${id}`),
  },
}
