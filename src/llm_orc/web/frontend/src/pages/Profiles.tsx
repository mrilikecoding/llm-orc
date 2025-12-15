import { signal } from '@preact/signals'
import { useEffect } from 'preact/hooks'
import { api, Profile, CreateProfileInput } from '../api/client'

const profiles = signal<Profile[]>([])
const loading = signal(true)
const error = signal<string | null>(null)
const showForm = signal(false)
const editingProfile = signal<Profile | null>(null)
const saving = signal(false)
const formError = signal<string | null>(null)

// Form fields
const formName = signal('')
const formProvider = signal('ollama')
const formModel = signal('')
const formSystemPrompt = signal('')
const formTimeout = signal('')

const PROVIDERS = ['ollama', 'anthropic', 'google', 'openai']

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

function resetForm() {
  formName.value = ''
  formProvider.value = 'ollama'
  formModel.value = ''
  formSystemPrompt.value = ''
  formTimeout.value = ''
  formError.value = null
}

function openCreateForm() {
  editingProfile.value = null
  resetForm()
  showForm.value = true
}

function openEditForm(profile: Profile) {
  editingProfile.value = profile
  formName.value = profile.name
  formProvider.value = profile.provider
  formModel.value = profile.model
  formSystemPrompt.value = profile.system_prompt || ''
  formTimeout.value = profile.timeout_seconds?.toString() || ''
  formError.value = null
  showForm.value = true
}

function closeForm() {
  showForm.value = false
  editingProfile.value = null
  resetForm()
}

async function handleSubmit(e: Event) {
  e.preventDefault()
  if (!formName.value.trim() || !formModel.value.trim()) {
    formError.value = 'Name and model are required'
    return
  }

  saving.value = true
  formError.value = null

  try {
    const input: CreateProfileInput = {
      name: formName.value.trim(),
      provider: formProvider.value,
      model: formModel.value.trim(),
      system_prompt: formSystemPrompt.value.trim() || undefined,
      timeout_seconds: formTimeout.value ? parseInt(formTimeout.value) : undefined,
    }

    if (editingProfile.value) {
      await api.profiles.update(editingProfile.value.name, {
        provider: input.provider,
        model: input.model,
        system_prompt: input.system_prompt,
        timeout_seconds: input.timeout_seconds,
      })
    } else {
      await api.profiles.create(input)
    }

    closeForm()
    await loadProfiles()
  } catch (e) {
    formError.value = e instanceof Error ? e.message : 'Failed to save profile'
  } finally {
    saving.value = false
  }
}

async function handleDelete(profile: Profile) {
  if (!confirm(`Delete profile "${profile.name}"?`)) return

  try {
    await api.profiles.delete(profile.name)
    await loadProfiles()
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to delete profile'
  }
}

function ProfileForm() {
  const isEditing = editingProfile.value !== null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-bg-secondary border border-border rounded-lg w-full max-w-md mx-4">
        <div className="p-4 border-b border-border flex justify-between items-center">
          <h3 className="text-lg font-semibold">
            {isEditing ? 'Edit Profile' : 'Create Profile'}
          </h3>
          <button
            className="text-text-secondary hover:text-text-primary text-xl"
            onClick={closeForm}
          >
            ×
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-4">
          {formError.value && (
            <div className="mb-4 p-3 bg-error-bg/20 border border-error rounded text-error text-sm">
              {formError.value}
            </div>
          )}

          <div className="mb-4">
            <label className="block text-sm text-text-secondary mb-1">Name</label>
            <input
              type="text"
              value={formName.value}
              onInput={(e) => (formName.value = (e.target as HTMLInputElement).value)}
              disabled={isEditing}
              className="w-full p-2 bg-bg-primary border border-border rounded text-text-primary
                focus:outline-none focus:border-accent disabled:opacity-50"
              placeholder="my-profile"
            />
          </div>

          <div className="mb-4">
            <label className="block text-sm text-text-secondary mb-1">Provider</label>
            <select
              value={formProvider.value}
              onChange={(e) => (formProvider.value = (e.target as HTMLSelectElement).value)}
              className="w-full p-2 bg-bg-primary border border-border rounded text-text-primary
                focus:outline-none focus:border-accent"
            >
              {PROVIDERS.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>

          <div className="mb-4">
            <label className="block text-sm text-text-secondary mb-1">Model</label>
            <input
              type="text"
              value={formModel.value}
              onInput={(e) => (formModel.value = (e.target as HTMLInputElement).value)}
              className="w-full p-2 bg-bg-primary border border-border rounded text-text-primary
                focus:outline-none focus:border-accent"
              placeholder="llama3.2:3b"
            />
          </div>

          <div className="mb-4">
            <label className="block text-sm text-text-secondary mb-1">
              System Prompt <span className="text-text-muted">(optional)</span>
            </label>
            <textarea
              value={formSystemPrompt.value}
              onInput={(e) => (formSystemPrompt.value = (e.target as HTMLTextAreaElement).value)}
              className="w-full p-2 bg-bg-primary border border-border rounded text-text-primary
                focus:outline-none focus:border-accent resize-y"
              rows={3}
              placeholder="You are a helpful assistant..."
            />
          </div>

          <div className="mb-6">
            <label className="block text-sm text-text-secondary mb-1">
              Timeout (seconds) <span className="text-text-muted">(optional)</span>
            </label>
            <input
              type="number"
              value={formTimeout.value}
              onInput={(e) => (formTimeout.value = (e.target as HTMLInputElement).value)}
              className="w-full p-2 bg-bg-primary border border-border rounded text-text-primary
                focus:outline-none focus:border-accent"
              placeholder="60"
            />
          </div>

          <div className="flex gap-3 justify-end">
            <button
              type="button"
              onClick={closeForm}
              className="px-4 py-2 text-text-secondary hover:text-text-primary"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={saving.value}
              className="px-4 py-2 bg-success-bg hover:bg-success-bg/80 text-white rounded
                disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {saving.value ? 'Saving...' : isEditing ? 'Update' : 'Create'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
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
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Model Profiles</h2>
        <button
          onClick={openCreateForm}
          className="px-4 py-2 bg-success-bg hover:bg-success-bg/80 text-white rounded font-medium"
        >
          + Create Profile
        </button>
      </div>

      {profiles.value.length === 0 ? (
        <div className="text-center p-12 text-text-secondary">
          <p>No profiles configured yet.</p>
          <p className="mt-2">Create a profile to get started.</p>
        </div>
      ) : (
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
              <th className="text-right py-3 px-4 bg-border-light border-b border-border font-semibold">
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {profiles.value.map((profile) => (
              <tr key={profile.name} className="hover:bg-border-light/30">
                <td className="py-3 px-4 border-b border-border">
                  <strong>{profile.name}</strong>
                </td>
                <td className="py-3 px-4 border-b border-border">
                  <span className="inline-block py-1 px-2 bg-border-light rounded text-sm">
                    {profile.provider}
                  </span>
                </td>
                <td className="py-3 px-4 border-b border-border font-mono text-sm">
                  {profile.model}
                </td>
                <td className="py-3 px-4 border-b border-border text-right">
                  <button
                    onClick={() => openEditForm(profile)}
                    className="text-accent hover:text-accent/80 mr-3"
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => handleDelete(profile)}
                    className="text-error hover:text-error/80"
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {showForm.value && <ProfileForm />}
    </div>
  )
}
