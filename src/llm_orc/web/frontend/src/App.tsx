import { signal } from '@preact/signals'
import { EnsemblesPage } from './pages/Ensembles'
import { ProfilesPage } from './pages/Profiles'
import { ArtifactsPage } from './pages/Artifacts'

type Page = 'ensembles' | 'profiles' | 'artifacts'

const currentPage = signal<Page>('ensembles')

function NavLink({ page, label }: { page: Page; label: string }) {
  const isActive = currentPage.value === page
  const baseClasses = 'block py-2 px-3 rounded-md text-text-primary cursor-pointer transition-colors'
  const activeClasses = isActive ? 'bg-border-light' : 'hover:bg-border-light/50'

  return (
    <li className="mb-2">
      <a
        className={`${baseClasses} ${activeClasses}`}
        onClick={() => (currentPage.value = page)}
      >
        {label}
      </a>
    </li>
  )
}

export function App() {
  return (
    <div className="flex min-h-screen">
      <aside className="w-[200px] bg-bg-secondary border-r border-border p-4">
        <div className="text-xl font-bold mb-6 text-accent">llm-orc</div>
        <nav>
          <ul className="list-none">
            <NavLink page="ensembles" label="Ensembles" />
            <NavLink page="profiles" label="Profiles" />
            <NavLink page="artifacts" label="Artifacts" />
          </ul>
        </nav>
      </aside>
      <main className="flex-1 p-6">
        {currentPage.value === 'ensembles' && <EnsemblesPage />}
        {currentPage.value === 'profiles' && <ProfilesPage />}
        {currentPage.value === 'artifacts' && <ArtifactsPage />}
      </main>
    </div>
  )
}
