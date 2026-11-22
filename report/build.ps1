# Build the report -> build/main.pdf
# Usage: .\build.ps1 [-Clean]
# Uses a local latexmk if available, otherwise the texlive/texlive Docker image.
param([switch]$Clean)

Set-Location $PSScriptRoot

if ($Clean) {
    if (Test-Path build) { Remove-Item -Recurse -Force build }
    Write-Host "Removed build/"
    exit 0
}

if (Get-Command latexmk -ErrorAction SilentlyContinue) {
    latexmk main.tex
} elseif (Get-Command docker -ErrorAction SilentlyContinue) {
    docker run --rm -v "${PWD}:/doc" -w /doc texlive/texlive:latest latexmk main.tex
} else {
    Write-Error "Neither latexmk nor docker found. Install TeX Live or Docker."
    exit 1
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Done: build/main.pdf"
