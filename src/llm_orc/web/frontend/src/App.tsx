import { signal } from '@preact/signals'
import { EnsemblesPage } from './pages/Ensembles'
import { ProfilesPage } from './pages/Profiles'
import { ScriptsPage } from './pages/Scripts'
import { ArtifactsPage } from './pages/Artifacts'

type Page = 'ensembles' | 'profiles' | 'scripts' | 'artifacts'

const currentPage = signal<Page>('ensembles')

export function App() {
  return (
    <>
      <header className="gradient-header">
        <h1>llm-orc</h1>
        <p>Multi-agent orchestration platform</p>
      </header>

      <nav className="tabs">
        <ul>
          {(['ensembles', 'profiles', 'scripts', 'artifacts'] as Page[]).map((page) => (
            <li key={page}>
              <button
                className={currentPage.value === page ? 'active' : ''}
                onClick={() => (currentPage.value = page)}
              >
                {page.charAt(0).toUpperCase() + page.slice(1)}
              </button>
            </li>
          ))}
        </ul>
      </nav>

      <main className="container">
        {currentPage.value === 'ensembles' && <EnsemblesPage />}
        {currentPage.value === 'profiles' && <ProfilesPage />}
        {currentPage.value === 'scripts' && <ScriptsPage />}
        {currentPage.value === 'artifacts' && <ArtifactsPage />}
      </main>
    </>
  )
}
