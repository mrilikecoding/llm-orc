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
    <div
      className={`bg-bg-secondary border rounded-lg p-4 cursor-pointer transition-all
        hover:shadow-lg hover:-translate-y-0.5
        ${isSelected ? 'border-accent ring-2 ring-accent/20' : 'border-border hover:border-text-secondary'}`}
      onClick={() => (selectedProfile.value = profile)}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="text-base font-semibold text-accent">{profile.name}</div>
        <span className="text-xs py-0.5 px-2 bg-border-light text-text-secondary rounded">
          {profile.provider}
        </span>
      </div>
      <div className="text-text-secondary text-sm font-mono">{profile.model}</div>
      {profile.system_prompt && (
        <div className="mt-2 text-xs text-text-muted line-clamp-2">
          {profile.system_prompt}
        </div>
      )}
    </div>
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
          {/* Model */}
          <div className="mb-4">
            <div className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
              Model
            </div>
            <code className="block p-3 bg-bg-primary border border-border rounded text-sm">
              {profile.model}
            </code>
          </div>

          {/* System Prompt */}
          {profile.system_prompt && (
            <div className="mb-4">
              <div className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
                System Prompt
              </div>
              <pre className="p-3 bg-bg-primary border border-border rounded text-sm whitespace-pre-wrap">
                {profile.system_prompt}
              </pre>
            </div>
          )}

          {/* Timeout */}
          {profile.timeout_seconds && (
            <div className="mb-4">
              <div className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
                Timeout
              </div>
              <div className="text-text-primary">{profile.timeout_seconds} seconds</div>
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-3 mt-6 pt-4 border-t border-border">
            <button
              onClick={() => openEditForm(profile)}
              className="flex-1 py-2.5 px-4 bg-accent hover:bg-accent/90 text-white rounded-lg font-medium transition-colors"
            >
              Edit Profile
            </button>
            <button
              onClick={() => handleDelete(profile)}
              className="py-2.5 px-4 border border-error text-error hover:bg-error/10 rounded-lg font-medium transition-colors"
            >
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
          <div className="mb-4 p-3 bg-error/10 border border-error/50 rounded-lg text-error text-sm">
            {formError.value}
          </div>
        )}

        <div className="mb-4">
          <label className="block text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
            Name
          </label>
          <input
            type="text"
            value={formName.value}
            onInput={(e) => (formName.value = (e.target as HTMLInputElement).value)}
            disabled={isEditing}
            className="w-full p-3 bg-bg-primary border border-border rounded-lg text-text-primary
              focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent/50
              disabled:opacity-50"
            placeholder="my-profile"
          />
        </div>

        <div className="mb-4">
          <label className="block text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
            Provider
          </label>
          <select
            value={formProvider.value}
            onChange={(e) => (formProvider.value = (e.target as HTMLSelectElement).value)}
            className="w-full p-3 bg-bg-primary border border-border rounded-lg text-text-primary
              focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent/50"
          >
            {PROVIDERS.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </div>

        <div className="mb-4">
          <label className="block text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
            Model
          </label>
          <input
            type="text"
            value={formModel.value}
            onInput={(e) => (formModel.value = (e.target as HTMLInputElement).value)}
            className="w-full p-3 bg-bg-primary border border-border rounded-lg text-text-primary
              focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent/50"
            placeholder="llama3.2:3b"
          />
        </div>

        <div className="mb-4">
          <label className="block text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
            System Prompt <span className="text-text-muted normal-case">(optional)</span>
          </label>
          <textarea
            value={formSystemPrompt.value}
            onInput={(e) => (formSystemPrompt.value = (e.target as HTMLTextAreaElement).value)}
            className="w-full p-3 bg-bg-primary border border-border rounded-lg text-text-primary
              focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent/50 resize-y"
            rows={3}
            placeholder="You are a helpful assistant..."
          />
        </div>

        <div className="mb-6">
          <label className="block text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
            Timeout (seconds) <span className="text-text-muted normal-case">(optional)</span>
          </label>
          <input
            type="number"
            value={formTimeout.value}
            onInput={(e) => (formTimeout.value = (e.target as HTMLInputElement).value)}
            className="w-full p-3 bg-bg-primary border border-border rounded-lg text-text-primary
              focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent/50"
            placeholder="60"
          />
        </div>

        <div className="flex gap-3">
          <button
            type="button"
            onClick={closeForm}
            className="flex-1 py-2.5 px-4 border border-border text-text-secondary hover:text-text-primary
              hover:border-text-secondary rounded-lg font-medium transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={saving.value}
            className={`flex-1 py-2.5 px-4 rounded-lg text-white font-medium transition-colors
              ${saving.value
                ? 'bg-border-light cursor-not-allowed'
                : 'bg-success-bg hover:bg-success-bg/90'}`}
          >
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
    return <div className="text-text-secondary">Loading profiles...</div>
  }

  if (error.value) {
    return <div className="text-error">Error: {error.value}</div>
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold">Model Profiles</h1>
          <p className="text-text-secondary text-sm mt-1">
            {profiles.value.length} profile{profiles.value.length !== 1 ? 's' : ''} configured
          </p>
        </div>
        <button
          onClick={openCreateForm}
          className="px-4 py-2.5 bg-success-bg hover:bg-success-bg/90 text-white rounded-lg font-medium transition-colors"
        >
          + Create Profile
        </button>
      </div>

      {profiles.value.length === 0 ? (
        <div className="text-center py-16 text-text-secondary">
          <div className="text-4xl mb-4 opacity-50">🤖</div>
          <p>No profiles configured yet.</p>
          <p className="text-sm mt-1">Create a profile to get started.</p>
        </div>
      ) : (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-4">
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
