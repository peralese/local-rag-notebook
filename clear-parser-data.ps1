<#
.SYNOPSIS
  Clears generated artifacts for local-rag-notebook (indexes, logs, caches).
.DESCRIPTION
  By default removes: index_store, logs, common caches.
  Use -Data to also delete ./data (original docs) — disabled by default.
.PARAMETER All
  Remove indexes, logs, caches, and data.
.PARAMETER Index
  Remove index_store.
.PARAMETER Logs
  Remove logs.
.PARAMETER Cache
  Remove caches (__pycache__, .pytest_cache, etc.).
.PARAMETER Data
  Remove data directory (original docs) — requires -Confirm or -Force.
.PARAMETER Force
  Skip confirmation prompts.
.PARAMETER DryRun
  Show what would be deleted without deleting.
.EXAMPLE
  .\clear-parser-data.ps1                  # (default) remove index_store + logs + caches
.EXAMPLE
  .\clear-parser-data.ps1 -Index -Logs     # only index + logs
.EXAMPLE
  .\clear-parser-data.ps1 -All -Force      # nuke everything including ./data
#>

param(
  [switch]$All,
  [switch]$Index,
  [switch]$Logs,
  [switch]$Cache,
  [switch]$Data,
  [switch]$Force,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Remove-Target {
  param(
    [string]$RelPath,
    [switch]$IsGlob
  )
  $full = Join-Path -Path $root -ChildPath $RelPath
  if ($IsGlob) {
    $items = Get-ChildItem -Path $full -Force -ErrorAction SilentlyContinue
    foreach ($it in $items) {
      if ($DryRun) { Write-Host "[DRYRUN] Would remove $($it.FullName)" -ForegroundColor Yellow }
      else {
        Write-Host "Removing $($it.FullName)" -ForegroundColor Yellow
        Remove-Item -LiteralPath $it.FullName -Recurse -Force -ErrorAction SilentlyContinue
      }
    }
  } else {
    if (Test-Path -LiteralPath $full) {
      if ($DryRun) { Write-Host "[DRYRUN] Would remove $full" -ForegroundColor Yellow }
      else {
        Write-Host "Removing $full" -ForegroundColor Yellow
        Remove-Item -LiteralPath $full -Recurse -Force -ErrorAction SilentlyContinue
      }
    } else {
      Write-Host "Skip $full (not found)" -ForegroundColor DarkGray
    }
  }
}

# Defaults: if no explicit flags and not -All, assume Index+Logs+Cache
if (-not ($All -or $Index -or $Logs -or $Cache -or $Data)) {
  $Index = $true; $Logs = $true; $Cache = $true
}

if ($All) { $Index = $true; $Logs = $true; $Cache = $true; $Data = $true }

# Safety check for Data
if ($Data -and -not $Force) {
  $ans = Read-Host "You are about to DELETE the ./data folder (original docs). Type 'YES' to continue"
  if ($ans -ne "YES") {
    Write-Host "Aborted by user." -ForegroundColor Red
    exit 1
  }
}

Write-Host "local-rag-notebook cleanup starting in $root" -ForegroundColor Cyan

if ($Index) { Remove-Target -RelPath "index_store" }
if ($Logs)  { Remove-Target -RelPath "logs" }

if ($Cache) {
  # Common caches
  Remove-Target -RelPath "__pycache__" -IsGlob
  Remove-Target -RelPath ".pytest_cache"
  Remove-Target -RelPath ".mypy_cache"
  Remove-Target -RelPath ".ruff_cache"
  # Project-wide __pycache__ folders
  $pycaches = Get-ChildItem -Path $root -Recurse -Directory -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq "__pycache__" }
  foreach ($d in $pycaches) {
    if ($DryRun) { Write-Host "[DRYRUN] Would remove $($d.FullName)" -ForegroundColor Yellow }
    else {
      Write-Host "Removing $($d.FullName)" -ForegroundColor Yellow
      Remove-Item -LiteralPath $d.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
  }
}

if ($Data) { Remove-Target -RelPath "data" }

Write-Host "Cleanup complete." -ForegroundColor Green
