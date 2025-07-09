# Homebrew Tap for LLM Orchestra

This directory contains the Homebrew formula for LLM Orchestra, ready to be used in the `mrilikecoding/llm-orchestra` tap.

## Creating the Tap Repository

To create the Homebrew tap, follow these steps:

1. **Create the tap repository on GitHub:**
   ```bash
   # Repository name MUST be: homebrew-llm-orchestra
   # GitHub URL will be: https://github.com/mrilikecoding/homebrew-llm-orchestra
   ```

2. **Initialize the tap repository:**
   ```bash
   git clone https://github.com/mrilikecoding/homebrew-llm-orchestra
   cd homebrew-llm-orchestra
   mkdir -p Formula
   ```

3. **Copy the formula:**
   ```bash
   cp /path/to/llm-orc/Formula/llm-orchestra.rb Formula/
   ```

4. **Create tap README:**
   ```bash
   cat > README.md << 'EOF'
   # LLM Orchestra Homebrew Tap

   This tap provides the LLM Orchestra formula for Homebrew.

   ## Installation

   ```bash
   brew tap mrilikecoding/llm-orchestra
   brew install llm-orchestra
   ```

   ## Usage

   After installation, you can use the `llm-orc` command:

   ```bash
   llm-orc --help
   llm-orc auth setup
   llm-orc list-ensembles
   ```

   ## About

   LLM Orchestra is a multi-agent LLM communication system with ensemble orchestration.
   
   - **Homepage**: https://github.com/mrilikecoding/llm-orc
   - **Documentation**: https://github.com/mrilikecoding/llm-orc#readme
   EOF
   ```

5. **Commit and push:**
   ```bash
   git add .
   git commit -m "feat: Add llm-orchestra formula v0.2.0"
   git push origin main
   ```

## Usage for End Users

Once the tap is published, users can install LLM Orchestra with:

```bash
brew tap mrilikecoding/llm-orchestra
brew install llm-orchestra
```

## Updating the Formula

When a new version is released:

1. Update the `url` to point to the new release tag
2. Update the `sha256` hash:
   ```bash
   curl -L -s "https://github.com/mrilikecoding/llm-orc/archive/refs/tags/vX.Y.Z.tar.gz" | shasum -a 256
   ```
3. Update the version in the URL
4. Commit and push the changes

## Testing the Formula

Before publishing, test the formula locally:

```bash
brew install --build-from-source ./Formula/llm-orchestra.rb
brew test llm-orchestra
```