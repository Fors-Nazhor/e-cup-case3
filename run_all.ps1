# Full solution, end to end, unattended.
#
# Stage order is deliberate: the LightGBM-only fit at stage 3 puts a valid
# submission.csv on disk within ~10 minutes, so every later stage is upside
# rather than a prerequisite. Stages run one at a time -- running the boosters
# next to the GPU job once crashed LightGBM with an access violation.
#
# Set CASE3_WORK to keep a second feature version side by side; features land in
# that directory and predictions in out_<dir>, so versions never mix.

param([string]$Work = "work")

$root = $PSScriptRoot
$py   = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$gpu  = "$root\.venv-rocm\Scripts\python.exe"     # ROCm build, see README

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:CASE3_WORK       = $Work
Set-Location "$root\src"
New-Item -ItemType Directory -Force -Path "$root\logs" | Out-Null

$main = "$root\logs\pipeline.log"
function Log($m) {
    $line = "[{0}] {1}`r`n" -f (Get-Date -Format 'HH:mm:ss'), $m
    try { [System.IO.File]::AppendAllText($main, $line, [System.Text.Encoding]::UTF8) } catch { }
    Write-Host $line.TrimEnd()
}
function Step($name, $exe, [string[]]$argv) {
    Log "START  $name"
    $t = Get-Date
    & $exe $argv *>> "$root\logs\$name.log"
    $code = $LASTEXITCODE
    $mins = [math]::Round(((Get-Date) - $t).TotalMinutes, 1)
    if ($code -eq 0) { Log "OK     $name  ($mins min)" }
    else             { Log "FAILED $name  exit=$code  ($mins min) -- continuing" }
}

Log "=== pipeline start (work dir: $Work) ==="

if (-not (Test-Path "$root\$Worknchor_2026-02-13.parquet")) {
    Step "1_features" $py @("build_features.py","--n-folds","15","--stride","14","--offsets","0,5,9")
}
if (-not (Test-Path "$root\work\daily.npy")) { Step "2_daily" $py @("build_daily.py") }
Step "4_val_gbdt"    $py  @("train.py","--mode","val","--models","lgb,cat,two","--rounds","6000",
                            "--anchor-step","3","--tau","0","--deseason","1")
Step "5_val_nn"      $gpu @("nn_train.py","--mode","val","--epochs","6","--anchor-step","3","--batch","4096")
Step "6_final_gbdt"  $py  @("train.py","--mode","final","--models","lgb,cat,two","--anchor-step","3",
                            "--tau","0","--deseason","1","--seeds","3","--final-rounds","auto")
Step "7_final_nn"    $gpu @("nn_train.py","--mode","final","--epochs","6","--anchor-step","3","--batch","4096")
Step "8_blend"       $py  @("blend.py")
Step "9_check"       $py  @("check_submission.py")

Log "=== pipeline done ==="
