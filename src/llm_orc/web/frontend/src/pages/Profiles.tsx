import { signal } from '@preact/signals'
import { useEffect } from 'preact/hooks'
import { api, Profile } from '../api/client'

const profiles = signal<Profile[]>([])
const loading = signal(true)
const error = signal<string | null>(null)

const styles = {
  title: {
    fontSize: '1.5rem',
    fontWeight: 'bold',
    marginBottom: '1.5rem',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse' as const,
    background: '#161b22',
    borderRadius: '8px',
    overflow: 'hidden',
  },
  th: {
    textAlign: 'left' as const,
    padding: '0.75rem 1rem',
    background: '#21262d',
    borderBottom: '1px solid #30363d',
    fontWeight: '600',
  },
  td: {
    padding: '0.75rem 1rem',
    borderBottom: '1px solid #30363d',
  },
  provider: {
    display: 'inline-block',
    padding: '0.25rem 0.5rem',
    background: '#21262d',
    borderRadius: '4px',
    fontSize: '0.85rem',
  },
}

async function loadProfiles() {
  loading.value = true
  error.value = null
  try {
    profiles.value = await api.profiles.list()
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load profiles'
  } finally {
    loading.value = false
  }
}

export function ProfilesPage() {
  useEffect(() => {
    loadProfiles()
  }, [])

  if (loading.value) {
    return <div>Loading profiles...</div>
  }

  if (error.value) {
    return <div style={{ color: '#f85149' }}>Error: {error.value}</div>
  }

  return (
    <div>
      <h2 style={styles.title}>Model Profiles</h2>

      <table style={styles.table}>
        <thead>
          <tr>
            <th style={styles.th}>Name</th>
            <th style={styles.th}>Provider</th>
            <th style={styles.th}>Model</th>
          </tr>
        </thead>
        <tbody>
          {profiles.value.map((profile) => (
            <tr key={profile.name}>
              <td style={styles.td}>
                <strong>{profile.name}</strong>
              </td>
              <td style={styles.td}>
                <span style={styles.provider}>{profile.provider}</span>
              </td>
              <td style={styles.td}>{profile.model}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
