const BASE_URL = '/api'

export interface Ensemble {
  name: string
  description: string
  source: string
  agents?: { name: string; model_profile: string }[]
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
    get: (name: string) => fetchJson<Ensemble>(`${BASE_URL}/ensembles/${name}`),
    execute: (name: string, input: string) =>
      fetchJson<{ status: string; results: Record<string, unknown> }>(
        `${BASE_URL}/ensembles/${name}/execute`,
        { method: 'POST', body: JSON.stringify({ input }) }
      ),
  },
  profiles: {
    list: () => fetchJson<Profile[]>(`${BASE_URL}/profiles`),
  },
  artifacts: {
    list: () => fetchJson<Artifact[]>(`${BASE_URL}/artifacts`),
  },
}
