#!/bin/bash
# LLM Orchestra Security Check Script
# Verifies that credential files are properly protected

echo "🔒 LLM Orchestra Security Check"
echo "================================"

# Check if gitignore properly excludes credential files
echo "Checking gitignore protection..."
if git check-ignore .llm-orc/credentials.enc >/dev/null 2>&1; then
    echo "✅ credentials.enc is gitignored"
else
    echo "❌ WARNING: credentials.enc is NOT gitignored!"
    echo "   Add '*.enc' and 'credentials.enc' to .gitignore"
fi

if git check-ignore .llm-orc/.key >/dev/null 2>&1; then
    echo "✅ .key is gitignored"  
else
    echo "❌ WARNING: .key is NOT gitignored!"
    echo "   Add '.key' to .gitignore"
fi

# Check local ensemble patterns
echo ""
echo "Checking local ensemble protection..."
if git check-ignore .llm-orc/ensembles/test-local.yaml >/dev/null 2>&1; then
    echo "✅ *-local.yaml patterns are gitignored"
else
    echo "❌ WARNING: Local ensemble patterns are NOT gitignored!"
    echo "   Add '*-local.yaml' and 'local-*.yaml' to .gitignore"
fi

if git check-ignore .llm-orc/ensembles/local-test.yaml >/dev/null 2>&1; then
    echo "✅ local-*.yaml patterns are gitignored"
else
    echo "❌ WARNING: Local ensemble patterns are NOT gitignored!"
    echo "   Add 'local-*.yaml' to .gitignore"
fi

# Check if any credential files are staged
echo ""
echo "Checking for staged credential files..."
if git diff --cached --name-only | grep -E "\.(enc|key)$|credentials\." >/dev/null; then
    echo "❌ WARNING: Credential files are staged for commit!"
    echo "   Run: git reset HEAD <filename> to unstage"
    git diff --cached --name-only | grep -E "\.(enc|key)$|credentials\."
else
    echo "✅ No credential files staged for commit"
fi

# Check file permissions if files exist
echo ""
echo "Checking credential file permissions..."
if [ -f ~/.llm-orc/credentials.enc ]; then
    perms=$(stat -f "%OLp" ~/.llm-orc/credentials.enc 2>/dev/null || stat -c "%a" ~/.llm-orc/credentials.enc 2>/dev/null)
    if [ "$perms" = "600" ]; then
        echo "✅ credentials.enc has secure permissions (600)"
    else
        echo "❌ WARNING: credentials.enc permissions are $perms (should be 600)"
        echo "   Run: chmod 600 ~/.llm-orc/credentials.enc"
    fi
else
    echo "ℹ️  No credentials.enc file found (normal for new installations)"
fi

if [ -f ~/.llm-orc/.key ]; then
    perms=$(stat -f "%OLp" ~/.llm-orc/.key 2>/dev/null || stat -c "%a" ~/.llm-orc/.key 2>/dev/null)
    if [ "$perms" = "600" ]; then
        echo "✅ .key has secure permissions (600)"
    else
        echo "❌ WARNING: .key permissions are $perms (should be 600)"
        echo "   Run: chmod 600 ~/.llm-orc/.key"
    fi
else
    echo "ℹ️  No .key file found (normal for new installations)"
fi

echo ""
echo "Security check complete!"
echo ""
echo "🛡️  Security Best Practices:"
echo "   • Never share credential files"
echo "   • Use *-local.yaml for personal experiments"
echo "   • Run this script before commits"
echo "   • Use 'git status' to verify no credential files are tracked"
echo "   • Credential files are automatically encrypted by LLM Orchestra"