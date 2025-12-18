import { signal } from '@preact/signals'
import { EnsemblesPage } from './pages/Ensembles'
import { ProfilesPage } from './pages/Profiles'
import { ScriptsPage } from './pages/Scripts'
import { ArtifactsPage } from './pages/Artifacts'

type Page = 'ensembles' | 'profiles' | 'scripts' | 'artifacts'

const currentPage = signal<Page>('ensembles')

function NavTab({ page, label }: { page: Page; label: string }) {
  const isActive = currentPage.value === page

  return (
    <button
      className={`px-6 py-3.5 font-semibold transition-all border-b-2 -mb-[2px]
        ${isActive
          ? 'text-accent border-accent bg-accent/10'
          : 'text-text-secondary border-transparent hover:text-text-primary hover:bg-white/5'}`}
      onClick={() => (currentPage.value = page)}
    >
      {label}
    </button>
  )
}

export function App() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Gradient header */}
      <header className="gradient-header px-6 py-5 shadow-lg">
        <h1 className="text-2xl font-bold m-0">llm-orc</h1>
        <p className="text-white/80 text-sm mt-1 m-0">Multi-agent orchestration platform</p>
      </header>

      {/* Navigation tabs */}
      <nav className="bg-bg-secondary border-b-2 border-border flex px-4">
        <NavTab page="ensembles" label="Ensembles" />
        <NavTab page="profiles" label="Profiles" />
        <NavTab page="scripts" label="Scripts" />
        <NavTab page="artifacts" label="Artifacts" />
      </nav>

      {/* Main content */}
      <main className="flex-1 p-8 overflow-auto">
        {currentPage.value === 'ensembles' && <EnsemblesPage />}
        {currentPage.value === 'profiles' && <ProfilesPage />}
        {currentPage.value === 'scripts' && <ScriptsPage />}
        {currentPage.value === 'artifacts' && <ArtifactsPage />}
      </main>
    </div>
  )
}
