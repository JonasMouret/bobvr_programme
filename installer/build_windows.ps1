<#
.SYNOPSIS
    Construit BobVr en exécutable Windows.

.DESCRIPTION
    À lancer depuis la racine du projet, dans PowerShell :

        .\installer\build_windows.ps1

    Le script crée son propre environnement virtuel, installe les dépendances,
    lance PyInstaller et vérifie le résultat. Il ne touche à rien d'autre sur
    la machine.

.PARAMETER Zip
    Produit en plus une archive dist\BobVr-<version>-win64.zip, prête à copier
    sur le poste de la piste.

.PARAMETER SkipVenv
    Utilise l'environnement Python courant au lieu d'en créer un.
#>

[CmdletBinding()]
param(
    [switch]$Zip,
    [switch]$SkipVenv
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Write-Host "Projet : $root" -ForegroundColor Cyan

# --- Python ---------------------------------------------------------------

$python = "python"
$version = & $python -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ([version]$version -lt [version]"3.11") {
    throw "Python $version détecté ; BobVr demande 3.11 ou plus récent."
}
Write-Host "Python $version" -ForegroundColor Cyan

# --- environnement --------------------------------------------------------

if (-not $SkipVenv) {
    $venv = Join-Path $root ".venv-build"
    if (-not (Test-Path $venv)) {
        Write-Host "Création de l'environnement $venv…"
        & $python -m venv $venv
    }
    $python = Join-Path $venv "Scripts\python.exe"
}

& $python -m pip install --upgrade pip --quiet
& $python -m pip install --upgrade -e ".[build]" --quiet
if ($LASTEXITCODE -ne 0) { throw "L'installation des dépendances a échoué." }

# --- outils embarqués -----------------------------------------------------

$vendor = Join-Path $root "installer\vendor"
$bundled = @(Get-ChildItem -Path $vendor -Filter *.exe -ErrorAction SilentlyContinue)
if ($bundled.Count -eq 0) {
    Write-Warning @"
Aucun exécutable dans installer\vendor : la build ne contiendra ni ffmpeg ni
exiftool. Elle fonctionnera seulement si ces outils sont sur le PATH du poste
qui l'utilisera. Voir docs\windows.md pour quelles versions déposer.
"@
} else {
    Write-Host "Outils embarqués : $($bundled.Name -join ', ')" -ForegroundColor Cyan
}

# --- build ----------------------------------------------------------------

Write-Host "PyInstaller…" -ForegroundColor Cyan
& $python -m PyInstaller "installer\bobvr.spec" --noconfirm --clean
if ($LASTEXITCODE -ne 0) { throw "PyInstaller a échoué." }

$out = Join-Path $root "dist\BobVr"
foreach ($required in @("BobVr.exe", "bobvr-cli.exe",
                        "_internal\bobvr\render\kernels\gopromax_equirect.cl")) {
    if (-not (Test-Path (Join-Path $out $required))) {
        throw "Build incomplète : $required est absent."
    }
}

$size = "{0:N0} Mo" -f ((Get-ChildItem $out -Recurse -File |
    Measure-Object -Property Length -Sum).Sum / 1MB)
Write-Host ""
Write-Host "Build terminée : $out ($size)" -ForegroundColor Green

# --- vérification ---------------------------------------------------------

Write-Host ""
Write-Host "Accélération vue par la build :" -ForegroundColor Cyan
& (Join-Path $out "bobvr-cli.exe") info

# --- archive --------------------------------------------------------------

if ($Zip) {
    $ver = & $python -c "import bobvr; print(bobvr.__version__)"
    $archive = Join-Path $root "dist\BobVr-$ver-win64.zip"
    if (Test-Path $archive) { Remove-Item $archive }
    Compress-Archive -Path $out -DestinationPath $archive
    Write-Host "Archive : $archive" -ForegroundColor Green
}

Write-Host ""
Write-Host "Prochaine étape : lancez $out\BobVr.exe, puis nommez les cartes" -ForegroundColor Yellow
Write-Host "avec bobvr-cli.exe label (voir docs\windows.md)." -ForegroundColor Yellow
