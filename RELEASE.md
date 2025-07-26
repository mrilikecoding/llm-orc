# Release Process

This document describes the automated release process for LLM Orchestra.

## Automated Homebrew Updates

The project includes a GitHub Actions workflow that automatically updates the Homebrew tap when a new release is published.

### How It Works

1. **Create a GitHub release** with a version tag (e.g., `v0.2.2`)
2. **GitHub Actions automatically triggers** the `update-homebrew.yml` workflow
3. **The workflow**:
   - Downloads the release tarball
   - Calculates the SHA256 hash
   - Updates the Homebrew formula with the new version and hash
   - Commits and pushes the changes to the tap repository
4. **Users can immediately install** the new version via Homebrew

### Setup Requirements

For the automation to work, you need:

1. **Personal Access Token**: Create a GitHub personal access token with `repo` permissions
2. **Add Secret**: Add the token as `HOMEBREW_TAP_TOKEN` in repository secrets
3. **Workflow File**: The `.github/workflows/update-homebrew.yml` file (already included)

### Creating a Release

To create a new release:

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.2.2"
   ```

2. **Update changelog** in `CHANGELOG.md`:
   ```markdown
   ## [0.2.2] - 2025-01-09
   ### Added
   - New feature description
   ### Fixed
   - Bug fix description
   ```

3. **Commit changes**:
   ```bash
   git add .
   git commit -m "chore: Bump version to 0.2.2 and update changelog"
   ```

4. **Create and push tag**:
   ```bash
   git tag -a v0.2.2 -m "Release v0.2.2: Description of changes"
   git push origin main
   git push origin v0.2.2
   ```

5. **Create GitHub release**:
   ```bash
   gh release create v0.2.2 --title "Release v0.2.2" --notes "Release notes here"
   ```

6. **Clean up lock file** (after automation completes):
   ```bash
   # Wait for workflows to complete, then commit any uv.lock changes
   # Note: uv.lock gets updated during CI/PyPI publishing but isn't included in release commit
   git add uv.lock
   git commit -m "chore: update uv.lock for version 0.2.2 release"
   git push origin main
   ```

7. **Automation takes over**:
   - The workflow will automatically update the Homebrew formula
   - Users can install the new version with `brew update && brew upgrade llm-orchestra`

### Manual Release Process (Backup)

If automation fails, you can manually update the Homebrew tap:

1. **Get the SHA256 hash**:
   ```bash
   curl -L -s "https://github.com/mrilikecoding/llm-orc/archive/refs/tags/v0.2.2.tar.gz" | shasum -a 256
   ```

2. **Update the formula** in the tap repository:
   ```ruby
   url "https://github.com/mrilikecoding/llm-orc/archive/refs/tags/v0.2.2.tar.gz"
   sha256 "new_hash_here"
   ```

3. **Commit and push**:
   ```bash
   git add Formula/llm-orchestra.rb
   git commit -m "feat: Update formula to v0.2.2"
   git push origin master
   ```

### Troubleshooting

If the automation fails:

1. **Check workflow logs** in GitHub Actions
2. **Verify the `HOMEBREW_TAP_TOKEN`** secret is set correctly
3. **Ensure the token has `repo` permissions**
4. **Check the tap repository** for any conflicts
5. **Fall back to manual release** if needed

### Release Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md` with release notes
- [ ] Commit and push changes
- [ ] Create and push git tag
- [ ] Create GitHub release
- [ ] Verify automation worked
- [ ] **Clean up lock file**: Commit any `uv.lock` changes after workflows complete
- [ ] Test installation: `brew update && brew upgrade llm-orchestra`
- [ ] Verify `llm-orc --version` shows correct version

## Benefits of Automated Releases

1. **Consistency**: Same process every time
2. **Speed**: Releases are available in Homebrew immediately
3. **Reliability**: Reduces manual errors
4. **Visibility**: Clear audit trail in GitHub Actions
5. **User Experience**: Users get updates faster

## Version Strategy

- **Major versions** (1.0.0): Breaking changes
- **Minor versions** (0.2.0): New features, backwards compatible
- **Patch versions** (0.2.1): Bug fixes, backwards compatible

Follow [Semantic Versioning](https://semver.org/) for version numbering.