import { signal } from '@preact/signals'
import { useEffect } from 'preact/hooks'
import { api, Profile, CreateProfileInput } from '../api/client'
import { SlidePanel } from '../components/SlidePanel'

const profiles = signal<Profile[]>([])
const loading = signal(true)
const error = signal<string | null>(null)
const showForm = signal(false)
const editingProfile = signal<Profile | null>(null)
const saving = signal(false)
const formError = signal<string | null>(null)
const selectedProfile = signal<Profile | null>(null)

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
    selectedProfile.value = null
    await loadProfiles()
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to delete profile'
  }
}

function ProfileCard({ profile }: { profile: Profile }) {
  const isSelected = selectedProfile.value?.name === profile.name

  return (
    <article
      className={`card${isSelected ? ' selected' : ''}`}
      onClick={() => (selectedProfile.value = profile)}
    >
      <header>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <strong>{profile.name}</strong>
          <span className="badge badge-primary">{profile.provider}</span>
        </div>
      </header>
      <p><code>{profile.model}</code></p>
      {profile.system_prompt && (
        <p><small>{profile.system_prompt}</small></p>
      )}
    </article>
  )
}

function ProfileDetailPanel() {
  const profile = selectedProfile.value

  return (
    <SlidePanel
      open={profile !== null}
      onClose={() => (selectedProfile.value = null)}
      title={profile?.name || ''}
      subtitle={`${profile?.provider} provider`}
      width="md"
    >
      {profile && (
        <>
          <div style={{ marginBottom: '1rem' }}>
            <p className="muted-label">Model</p>
            <code>{profile.model}</code>
          </div>

          {profile.system_prompt && (
            <div style={{ marginBottom: '1rem' }}>
              <p className="muted-label">System Prompt</p>
              <pre><code>{profile.system_prompt}</code></pre>
            </div>
          )}

          {profile.timeout_seconds && (
            <div style={{ marginBottom: '1rem' }}>
              <p className="muted-label">Timeout</p>
              <p>{profile.timeout_seconds} seconds</p>
            </div>
          )}

          <hr />
          <div role="group">
            <button onClick={() => openEditForm(profile)}>Edit Profile</button>
            <button className="outline secondary" onClick={() => handleDelete(profile)}>
              Delete
            </button>
          </div>
        </>
      )}
    </SlidePanel>
  )
}

function ProfileFormPanel() {
  const isEditing = editingProfile.value !== null

  return (
    <SlidePanel
      open={showForm.value}
      onClose={closeForm}
      title={isEditing ? 'Edit Profile' : 'Create Profile'}
      subtitle={isEditing ? `Editing ${editingProfile.value?.name}` : 'Add a new model profile'}
      width="md"
    >
      <form onSubmit={handleSubmit}>
        {formError.value && (
          <p style={{ color: '#f85149' }}>{formError.value}</p>
        )}

        <label>
          Name
          <input
            type="text"
            value={formName.value}
            onInput={(e) => (formName.value = (e.target as HTMLInputElement).value)}
            disabled={isEditing}
            placeholder="my-profile"
          />
        </label>

        <label>
          Provider
          <select
            value={formProvider.value}
            onChange={(e) => (formProvider.value = (e.target as HTMLSelectElement).value)}
          >
            {PROVIDERS.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </label>

        <label>
          Model
          <input
            type="text"
            value={formModel.value}
            onInput={(e) => (formModel.value = (e.target as HTMLInputElement).value)}
            placeholder="llama3.2:3b"
          />
        </label>

        <label>
          System Prompt <small>(optional)</small>
          <textarea
            value={formSystemPrompt.value}
            onInput={(e) => (formSystemPrompt.value = (e.target as HTMLTextAreaElement).value)}
            rows={3}
            placeholder="You are a helpful assistant..."
          />
        </label>

        <label>
          Timeout (seconds) <small>(optional)</small>
          <input
            type="number"
            value={formTimeout.value}
            onInput={(e) => (formTimeout.value = (e.target as HTMLInputElement).value)}
            placeholder="60"
          />
        </label>

        <div role="group">
          <button type="button" className="outline secondary" onClick={closeForm}>
            Cancel
          </button>
          <button type="submit" disabled={saving.value} aria-busy={saving.value}>
            {saving.value ? 'Saving...' : isEditing ? 'Update' : 'Create'}
          </button>
        </div>
      </form>
    </SlidePanel>
  )
}

export function ProfilesPage() {
  useEffect(() => {
    loadProfiles()
  }, [])

  if (loading.value) {
    return <p aria-busy="true">Loading profiles...</p>
  }

  if (error.value) {
    return <p style={{ color: '#f85149' }}>Error: {error.value}</p>
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h1>Model Profiles</h1>
          <p>{profiles.value.length} profile{profiles.value.length !== 1 ? 's' : ''} configured</p>
        </div>
        <button onClick={openCreateForm}>+ Create Profile</button>
      </div>

      {profiles.value.length === 0 ? (
        <div className="empty-state">
          <p>No profiles configured yet.</p>
          <p><small>Create a profile to get started.</small></p>
        </div>
      ) : (
        <div className="card-grid">
          {profiles.value.map((profile) => (
            <ProfileCard key={profile.name} profile={profile} />
          ))}
        </div>
      )}

      <ProfileDetailPanel />
      <ProfileFormPanel />
    </div>
  )
}
