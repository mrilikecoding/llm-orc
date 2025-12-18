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
  system_prompt?: string
  timeout_seconds?: number
}

export interface CreateProfileInput {
  name: string
  provider: string
  model: string
  system_prompt?: string
  timeout_seconds?: number
}

export interface UpdateProfileInput {
  provider?: string
  model?: string
  system_prompt?: string
  timeout_seconds?: number
}

export interface Script {
  name: string
  category: string
  path: string
}

export interface ScriptDetail {
  name: string
  category: string
  path: string
  content?: string
  description?: string
}

export interface ScriptTestResult {
  success: boolean
  output?: string
  error?: string
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
    create: (input: CreateProfileInput) =>
      fetchJson<Profile>(`${BASE_URL}/profiles`, {
        method: 'POST',
        body: JSON.stringify(input),
      }),
    update: (name: string, input: UpdateProfileInput) =>
      fetchJson<Profile>(`${BASE_URL}/profiles/${name}`, {
        method: 'PUT',
        body: JSON.stringify(input),
      }),
    delete: (name: string) =>
      fetchJson<{ deleted: boolean }>(`${BASE_URL}/profiles/${name}`, {
        method: 'DELETE',
      }),
  },
  scripts: {
    list: () => fetchJson<{ scripts: Script[] }>(`${BASE_URL}/scripts`),
    get: (category: string, name: string) =>
      fetchJson<ScriptDetail>(`${BASE_URL}/scripts/${category}/${name}`),
    test: (category: string, name: string, input: string) =>
      fetchJson<ScriptTestResult>(`${BASE_URL}/scripts/${category}/${name}/test`, {
        method: 'POST',
        body: JSON.stringify({ input }),
      }),
  },
  artifacts: {
    list: () => fetchJson<Artifact[]>(`${BASE_URL}/artifacts`),
    getForEnsemble: (ensemble: string) =>
      fetchJson<ArtifactDetail[]>(`${BASE_URL}/artifacts/${ensemble}`),
    get: (ensemble: string, id: string) =>
      fetchJson<ArtifactDetail>(`${BASE_URL}/artifacts/${ensemble}/${id}`),
  },
}
