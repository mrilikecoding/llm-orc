import { signal } from '@preact/signals'
import { EnsemblesPage } from './pages/Ensembles'
import { ProfilesPage } from './pages/Profiles'
import { ArtifactsPage } from './pages/Artifacts'

type Page = 'ensembles' | 'profiles' | 'artifacts'

const currentPage = signal<Page>('ensembles')

const styles = {
  container: {
    display: 'flex',
    minHeight: '100vh',
  },
  sidebar: {
    width: '200px',
    background: '#161b22',
    borderRight: '1px solid #30363d',
    padding: '1rem',
  },
  main: {
    flex: 1,
    padding: '1.5rem',
  },
  logo: {
    fontSize: '1.25rem',
    fontWeight: 'bold',
    marginBottom: '1.5rem',
    color: '#58a6ff',
  },
  nav: {
    listStyle: 'none',
  },
  navItem: {
    marginBottom: '0.5rem',
  },
  navLink: {
    display: 'block',
    padding: '0.5rem 0.75rem',
    borderRadius: '6px',
    color: '#c9d1d9',
    textDecoration: 'none',
    cursor: 'pointer',
    transition: 'background 0.2s',
  },
  navLinkActive: {
    background: '#21262d',
  },
}

function NavLink({ page, label }: { page: Page; label: string }) {
  const isActive = currentPage.value === page
  return (
    <li style={styles.navItem}>
      <a
        style={{
          ...styles.navLink,
          ...(isActive ? styles.navLinkActive : {}),
        }}
        onClick={() => (currentPage.value = page)}
      >
        {label}
      </a>
    </li>
  )
}

export function App() {
  return (
    <div style={styles.container}>
      <aside style={styles.sidebar}>
        <div style={styles.logo}>llm-orc</div>
        <nav>
          <ul style={styles.nav}>
            <NavLink page="ensembles" label="Ensembles" />
            <NavLink page="profiles" label="Profiles" />
            <NavLink page="artifacts" label="Artifacts" />
          </ul>
        </nav>
      </aside>
      <main style={styles.main}>
        {currentPage.value === 'ensembles' && <EnsemblesPage />}
        {currentPage.value === 'profiles' && <ProfilesPage />}
        {currentPage.value === 'artifacts' && <ArtifactsPage />}
      </main>
    </div>
  )
}
