# Where is the overnight pipeline right now?
$root = $PSScriptRoot

Write-Host ""
Write-Host "=== progress ===" -ForegroundColor Cyan
$log = Join-Path $root "logs\pipeline.log"
if (Test-Path $log) { Get-Content $log } else { Write-Host "pipeline has not started" }

Write-Host ""
Write-Host "=== running now ===" -ForegroundColor Cyan
$py = Get-Process python -ErrorAction SilentlyContinue
if ($py) {
    foreach ($p in $py) {
        $mins = [math]::Round(((Get-Date) - $p.StartTime).TotalMinutes, 1)
        $cores = if ($mins -gt 0) { [math]::Round($p.CPU / ($mins * 60), 1) } else { 0 }
        "  python PID {0}  {1} min  ~{2} cores  {3:N1} GB" -f $p.Id, $mins, $cores, ($p.WorkingSet64/1GB)
    }
} else { Write-Host "  no python running" }

$sh = Get-Process powershell -ErrorAction SilentlyContinue |
      Where-Object { $_.MainWindowTitle -eq "" -and $_.Id -ne $PID }
if (-not $py -and -not $sh) { Write-Host "  pipeline appears finished or stopped" }

Write-Host ""
Write-Host "=== last line of each stage ===" -ForegroundColor Cyan
Get-ChildItem (Join-Path $root "logs") -Filter "*.log" -ErrorAction SilentlyContinue |
  Where-Object { $_.BaseName -match '^\d+_' } | Sort-Object Name | ForEach-Object {
    $tail = (Get-Content $_.FullName -Tail 40 |
             Where-Object { $_ -notmatch "MIOpen|Warning|^\s*$" } | Select-Object -Last 1)
    "  {0,-22} {1}" -f $_.BaseName, $tail
  }

Write-Host ""
Write-Host "=== submission ===" -ForegroundColor Cyan
$sub = Join-Path $root "submission.csv"
if (Test-Path $sub) {
    $n = (Get-Content $sub | Measure-Object -Line).Lines
    $f = Get-Item $sub
    "  submission.csv  {0} lines  {1:N1} MB  written {2}" -f $n, ($f.Length/1MB), $f.LastWriteTime.ToString("HH:mm:ss")
    Write-Host "  first rows:"
    Get-Content $sub -TotalCount 4 | ForEach-Object { "    $_" }
} else { Write-Host "  not produced yet" }
Write-Host ""
