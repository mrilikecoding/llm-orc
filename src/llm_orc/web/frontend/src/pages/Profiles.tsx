import { signal } from '@preact/signals'
import { useEffect } from 'preact/hooks'
import { api, Profile } from '../api/client'

const profiles = signal<Profile[]>([])
const loading = signal(true)
const error = signal<string | null>(null)

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
    return <div className="text-error">Error: {error.value}</div>
  }

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Model Profiles</h2>

      <table className="w-full border-collapse bg-bg-secondary rounded-lg overflow-hidden">
        <thead>
          <tr>
            <th className="text-left py-3 px-4 bg-border-light border-b border-border font-semibold">
              Name
            </th>
            <th className="text-left py-3 px-4 bg-border-light border-b border-border font-semibold">
              Provider
            </th>
            <th className="text-left py-3 px-4 bg-border-light border-b border-border font-semibold">
              Model
            </th>
          </tr>
        </thead>
        <tbody>
          {profiles.value.map((profile) => (
            <tr key={profile.name}>
              <td className="py-3 px-4 border-b border-border">
                <strong>{profile.name}</strong>
              </td>
              <td className="py-3 px-4 border-b border-border">
                <span className="inline-block py-1 px-2 bg-border-light rounded text-sm">
                  {profile.provider}
                </span>
              </td>
              <td className="py-3 px-4 border-b border-border">{profile.model}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
