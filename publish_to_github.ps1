# VolveRAG - GitHub Publishing Script
# Run this script after restarting PowerShell to ensure Git is in PATH

Write-Host "=== VolveRAG GitHub Publishing Script ===" -ForegroundColor Cyan
Write-Host ""

# Check if git is available
try {
    $gitVersion = git --version 2>&1
    Write-Host "✓ Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git not found. Please restart PowerShell after installing Git." -ForegroundColor Red
    Write-Host "  Or install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Navigate to repository root
$repoRoot = "C:\Users\samue\Downloads\spwla_volve-main"
Set-Location $repoRoot
Write-Host "Working directory: $repoRoot" -ForegroundColor Cyan
Write-Host ""

# Step 1: Initialize git (if not already done)
if (Test-Path .git) {
    Write-Host "✓ Git repository already initialized" -ForegroundColor Green
} else {
    Write-Host "Initializing git repository..." -ForegroundColor Yellow
    git init
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Failed to initialize git repository" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Git repository initialized" -ForegroundColor Green
}

# Step 2: Add all files
Write-Host ""
Write-Host "Adding all files to git..." -ForegroundColor Yellow
git add .
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to add files" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Files added" -ForegroundColor Green

# Step 3: Check if there are changes to commit
$status = git status --porcelain
if (-not $status) {
    Write-Host ""
    Write-Host "No changes to commit. Checking if commit exists..." -ForegroundColor Yellow
    $commitExists = git log -1 --oneline 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Creating initial commit..." -ForegroundColor Yellow
        git commit -m "Initial commit: VolveRAG - Advanced RAG system for petrophysical reports"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "✗ Failed to create commit" -ForegroundColor Red
            exit 1
        }
        Write-Host "✓ Initial commit created" -ForegroundColor Green
    } else {
        Write-Host "✓ Commit already exists: $commitExists" -ForegroundColor Green
    }
} else {
    Write-Host ""
    Write-Host "Creating commit..." -ForegroundColor Yellow
    git commit -m "Initial commit: VolveRAG - Advanced RAG system for petrophysical reports"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Failed to create commit" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Commit created" -ForegroundColor Green
}

# Step 4: Check if remote already exists
Write-Host ""
$remoteExists = git remote get-url origin 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Remote 'origin' already exists: $remoteExists" -ForegroundColor Yellow
    $response = Read-Host "Do you want to remove and re-add it? (y/n)"
    if ($response -eq "y" -or $response -eq "Y") {
        git remote remove origin
        Write-Host "✓ Removed existing remote" -ForegroundColor Green
    } else {
        Write-Host "Keeping existing remote. Skipping remote add." -ForegroundColor Yellow
        $skipRemote = $true
    }
}

# Step 5: Add remote
if (-not $skipRemote) {
    Write-Host ""
    Write-Host "Adding remote repository..." -ForegroundColor Yellow
    git remote add origin https://github.com/samwiserr/VolveRAG.git
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Failed to add remote" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Remote added" -ForegroundColor Green
}

# Step 6: Rename branch to main
Write-Host ""
Write-Host "Renaming branch to main..." -ForegroundColor Yellow
git branch -M main
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ Warning: Branch rename may have failed (branch might already be 'main')" -ForegroundColor Yellow
} else {
    Write-Host "✓ Branch renamed to main" -ForegroundColor Green
}

# Step 7: Push to GitHub
Write-Host ""
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
Write-Host "Note: You may be prompted for GitHub credentials" -ForegroundColor Cyan
git push -u origin main
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "✗ Push failed. Common issues:" -ForegroundColor Red
    Write-Host "  1. Authentication required - use GitHub Personal Access Token" -ForegroundColor Yellow
    Write-Host "  2. Repository doesn't exist yet - create it at https://github.com/samwiserr/VolveRAG" -ForegroundColor Yellow
    Write-Host "  3. Network issues" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "=== SUCCESS! ===" -ForegroundColor Green
Write-Host "Your repository has been published to:" -ForegroundColor Cyan
Write-Host "https://github.com/samwiserr/VolveRAG" -ForegroundColor Cyan
Write-Host ""

