#!/bin/bash

# Script to prepare the repository for public sharing
# This script helps ensure no sensitive data is accidentally pushed to public repositories

echo "==== Preparing repository for public sharing ===="

# Check if .env files exist
ENV_FILES=$(find . -name ".env*" ! -name ".env.template")
if [ -n "$ENV_FILES" ]; then
  echo "⚠️  WARNING: Found .env files that should be removed or added to .gitignore:"
  echo "$ENV_FILES"
  
  read -p "Do you want to backup these files and replace with templates? (y/n) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    for env_file in $ENV_FILES; do
      # Create backup with .bak extension
      cp "$env_file" "${env_file}.bak"
      echo "✅ Backed up $env_file to ${env_file}.bak"
      
      # If .env.template exists, copy it to replace the .env file
      template_file="${env_file%.*}.template"
      if [ -f "$template_file" ]; then
        cp "$template_file" "$env_file"
        echo "✅ Replaced $env_file with its template version"
      else
        # If no template exists, create a blank file with a warning
        echo "# WARNING: This is a placeholder file. Do not store real credentials here." > "$env_file"
        echo "# Create proper credentials before running the application." >> "$env_file"
        echo "✅ Replaced $env_file with a placeholder"
      fi
    done
  fi
else
  echo "✅ No .env files found (good!)"
fi

# Check for potential private keys in code
echo -e "\nChecking for potential private keys in code..."
PRIVATE_KEY_PATTERN="private[_]?key|secret[_]?key|password|[0-9a-fA-F]{64}"
PRIVATE_KEY_FILES=$(grep -r --include="*.{js,py,ts,sol,json,yaml,yml}" -E "$PRIVATE_KEY_PATTERN" . | grep -v "node_modules" | grep -v ".git")

if [ -n "$PRIVATE_KEY_FILES" ]; then
  echo "⚠️  WARNING: Found potential private keys or secrets in code:"
  echo "$PRIVATE_KEY_FILES"
  echo "Please review these files and ensure they do not contain real secrets."
else
  echo "✅ No obvious private keys found in code"
fi

# Check blockchain configuration files
if [ -f "blockchain_config.json" ]; then
  echo -e "\nChecking blockchain_config.json..."
  if grep -q "0x[0-9a-fA-F]\{40\}" blockchain_config.json; then
    echo "⚠️  WARNING: blockchain_config.json contains what appear to be real contract addresses."
    echo "Consider replacing these with placeholder addresses before making public."
  else
    echo "✅ blockchain_config.json appears to be using placeholder addresses"
  fi
fi

# Check for large files that might be data or models
echo -e "\nChecking for large files that might contain data or models..."
LARGE_FILES=$(find . -type f -size +10M | grep -v "node_modules" | grep -v ".git")
if [ -n "$LARGE_FILES" ]; then
  echo "⚠️  WARNING: Found large files that might contain data or models:"
  echo "$LARGE_FILES"
  echo "Consider whether these should be excluded before making the repository public."
else
  echo "✅ No large files found"
fi

echo -e "\n==== Security check complete ===="
echo "Remember to review the .gitignore file to ensure all sensitive files are excluded."
echo "For additional security, consider running: git clean -fdx"
echo "This will remove all untracked files and directories (use with caution!)." 