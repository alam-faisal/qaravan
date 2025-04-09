#!/bin/bash

# Prompt for version
echo "🔢 Enter the new version (e.g. 0.1.3):"
read VERSION

# Prompt for commit message
echo "💬 Enter a descriptive commit message:"
read COMMIT_MSG

# 1. Update setup.py
echo "📦 Updating setup.py..."
sed -i "s/version=\"[0-9]\+\.[0-9]\+\.[0-9]\+\"/version=\"$VERSION\"/" setup.py

# 2. Update __init__.py
echo "📦 Updating __init__.py..."
sed -i "s/__version__ = \"[0-9]\+\.[0-9]\+\.[0-9]\+\"/__version__ = \"$VERSION\"/" src/qaravan/__init__.py

# 3. Commit changes
echo "📝 Committing changes..."
git add .
git commit -m "$COMMIT_MSG"
git push

# 4. Tag the release
echo "🏷️ Tagging release as v$VERSION"
git tag v$VERSION
git push origin v$VERSION

echo "✅ Done! Version $VERSION has been committed, pushed, and tagged."