# PowerShell script to install antiword on Windows
# Run this script as Administrator or it will install to user directory

Write-Host "Installing antiword for .doc file support..." -ForegroundColor Cyan

$installDir = "$env:USERPROFILE\antiword"
$zipPath = "$env:TEMP\antiword.zip"

# Create installation directory
New-Item -ItemType Directory -Force -Path $installDir | Out-Null

# Try multiple download sources
$downloadUrls = @(
    "https://github.com/rsdoiel/antiword/releases/download/v0.37/antiword-0.37.zip",
    "http://www.winfield.demon.nl/antiword-0.37.zip",
    "https://sourceforge.net/projects/antiword/files/antiword/antiword-0.37/antiword-0.37.zip/download"
)

$downloaded = $false
foreach ($url in $downloadUrls) {
    try {
        Write-Host "Trying to download from: $url" -ForegroundColor Yellow
        Invoke-WebRequest -Uri $url -OutFile $zipPath -ErrorAction Stop
        $downloaded = $true
        Write-Host "Download successful!" -ForegroundColor Green
        break
    } catch {
        Write-Host "Download failed: $_" -ForegroundColor Red
        continue
    }
}

if (-not $downloaded) {
    Write-Host "`nAutomatic download failed. Please download antiword manually:" -ForegroundColor Red
    Write-Host "1. Visit: http://www.winfield.demon.nl/" -ForegroundColor Yellow
    Write-Host "2. Download antiword-0.37.zip" -ForegroundColor Yellow
    Write-Host "3. Extract to: $installDir" -ForegroundColor Yellow
    Write-Host "4. Add to PATH: $installDir" -ForegroundColor Yellow
    exit 1
}

# Extract the zip file
Write-Host "Extracting antiword..." -ForegroundColor Cyan
try {
    Expand-Archive -Path $zipPath -DestinationPath $installDir -Force
    Write-Host "Extraction successful!" -ForegroundColor Green
} catch {
    Write-Host "Extraction failed: $_" -ForegroundColor Red
    exit 1
}

# Find the antiword.exe file
$antiwordExe = Get-ChildItem -Path $installDir -Recurse -Filter "antiword.exe" | Select-Object -First 1

if ($antiwordExe) {
    $antiwordDir = $antiwordExe.DirectoryName
    Write-Host "Found antiword.exe at: $antiwordDir" -ForegroundColor Green
    
    # Add to user PATH
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($currentPath -notlike "*$antiwordDir*") {
        [Environment]::SetEnvironmentVariable("Path", "$currentPath;$antiwordDir", "User")
        Write-Host "Added to user PATH: $antiwordDir" -ForegroundColor Green
    } else {
        Write-Host "Already in PATH" -ForegroundColor Yellow
    }
    
    # Test installation
    $env:Path += ";$antiwordDir"
    try {
        $version = & "$antiwordDir\antiword.exe" --version 2>&1
        Write-Host "`n✅ antiword installed successfully!" -ForegroundColor Green
        Write-Host "Version: $version" -ForegroundColor Cyan
        Write-Host "`nPlease restart your terminal/PowerShell for PATH changes to take effect." -ForegroundColor Yellow
    } catch {
        Write-Host "Installation complete but verification failed. Please restart terminal." -ForegroundColor Yellow
    }
} else {
    Write-Host "Could not find antiword.exe in extracted files." -ForegroundColor Red
    Write-Host "Please check the extraction directory: $installDir" -ForegroundColor Yellow
    exit 1
}

# Cleanup
Remove-Item $zipPath -ErrorAction SilentlyContinue

Write-Host "`nInstallation script completed!" -ForegroundColor Green





