#!/bin/bash
# Fix import paths in packages/core

cd /home/user/visionforge/packages/core/src

echo "Fixing import paths in packages/core..."

# Fix imports in components/ directory
echo "Fixing components/*.tsx..."
find components/ -name "*.tsx" -type f | while read file; do
    # @/components/ui/ → ./ui/ (same level imports)
    sed -i 's|@/components/ui/|./ui/|g' "$file"

    # @/lib/ → ../lib/ (up one level)
    sed -i 's|@/lib/|../lib/|g' "$file"

    # @/hooks/ → ../hooks/ (up one level)
    sed -i 's|@/hooks/|../hooks/|g' "$file"

    # @/components/ → ./ (same level)
    sed -i 's|@/components/|./|g' "$file"
done

# Fix imports in components/ui/ directory
echo "Fixing components/ui/*.tsx..."
find components/ui/ -name "*.tsx" -type f | while read file; do
    # @/components/ui/ → ./ (same directory)
    sed -i 's|@/components/ui/|./|g' "$file"

    # @/lib/ → ../../lib/ (up two levels)
    sed -i 's|@/lib/|../../lib/|g' "$file"

    # @/hooks/ → ../../hooks/ (up two levels)
    sed -i 's|@/hooks/|../../hooks/|g' "$file"

    # @/components/ → ../ (up one level)
    sed -i 's|@/components/|../|g' "$file"
done

# Fix imports in lib/ directory
echo "Fixing lib/*.ts..."
find lib/ -name "*.ts" -type f | while read file; do
    # @/lib/ → ./ (same directory)
    sed -i 's|@/lib/|./|g' "$file"

    # @/components/ → ../components/ (up one level)
    sed -i 's|@/components/|../components/|g' "$file"

    # @/hooks/ → ../hooks/ (up one level)
    sed -i 's|@/hooks/|../hooks/|g' "$file"
done

# Fix imports in hooks/ directory
echo "Fixing hooks/*.ts..."
find hooks/ -name "*.ts" -type f | while read file; do
    # @/hooks/ → ./ (same directory)
    sed -i 's|@/hooks/|./|g' "$file"

    # @/lib/ → ../lib/ (up one level)
    sed -i 's|@/lib/|../lib/|g' "$file"

    # @/components/ → ../components/ (up one level)
    sed -i 's|@/components/|../components/|g' "$file"
done

echo "✅ Import paths fixed!"
echo ""
echo "Verifying: Remaining @/ imports..."
grep -r "@/" . --include="*.ts" --include="*.tsx" | wc -l
