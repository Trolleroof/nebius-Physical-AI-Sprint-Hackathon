<#
.SYNOPSIS
    SO-101 sim pipeline on THIS Windows machine: collect -> build_dataset -> train.

.DESCRIPTION
    Stage 1 (collect) runs in sim\mujoco_venv (mujoco + numpy + imageio only).
    Stages 2 and 3 (build_dataset / lerobot-train) need cv2 + torch + lerobot,
    which do NOT live in mujoco_venv -- point -LerobotPython at an environment
    that has them, or pass -CollectOnly.

.EXAMPLE
    powershell -File sim\mujoco\run.ps1 -Episodes 60
    powershell -File sim\mujoco\run.ps1 -Episodes 60 -CollectOnly
#>
[CmdletBinding()]
param(
    [int]    $Episodes       = 60,
    [int]    $Seed           = 0,
    [string] $RawDir         = "data/sim_raw",
    [string] $DatasetRoot    = "data/lerobot/so101_pick_place",
    [string] $RepoId         = "local/so101_pick_place",
    [string] $OutputDir      = "outputs/act_so101",
    [string] $Scene          = "sim/mujoco/scene.xml",
    [ValidateSet("friction", "weld")]
    [string] $Grasp          = "friction",
    [int]    $Steps          = 8000,
    [string] $Device         = "cuda",
    [string] $Task           = "Put the orange cube in the tray.",
    [string] $LerobotPython  = "python",
    [switch] $WorldVideo,
    [switch] $KeepFailures,
    [switch] $CollectOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repoRoot
Write-Host "repo root: $repoRoot" -ForegroundColor Cyan

$SimPython = Join-Path $repoRoot "sim\mujoco_venv\Scripts\python.exe"
if (-not (Test-Path $SimPython)) { throw "missing sim venv python: $SimPython" }

# ------------------------------------------------------- 0. wrist lock gate
# wrist_roll's servo is broken on the physical arm and the joint is taped at
# -pi/2.  Prove it cannot move before spending an hour training on data that
# might roll it.
& $SimPython "sim/mujoco/check_wrist_lock.py" "--scene" $Scene
if ($LASTEXITCODE -ne 0) { throw "wrist lock check failed ($LASTEXITCODE) -- do not train" }

# ---------------------------------------------------------------- 1. collect
$collectArgs = @(
    "sim/mujoco/collect.py",
    "--episodes", $Episodes,
    "--out",      $RawDir,
    "--seed",     $Seed,
    "--scene",    $Scene,
    "--grasp",    $Grasp
)
if ($WorldVideo)   { $collectArgs += "--world-video" }
if ($KeepFailures) { $collectArgs += "--keep-failures" }

Write-Host "`n== 1/3 collect ==" -ForegroundColor Green
& $SimPython @collectArgs
if ($LASTEXITCODE -ne 0) { throw "collect.py failed ($LASTEXITCODE)" }

# Re-check against what was actually recorded, not just what the code intends.
& $SimPython "sim/mujoco/check_wrist_lock.py" "--scene" $Scene "--episodes" $RawDir
if ($LASTEXITCODE -ne 0) { throw "wrist lock check failed on recorded episodes -- do not train" }

if ($CollectOnly) { Write-Host "`n-CollectOnly: stopping after collection." -ForegroundColor Yellow; exit 0 }

# ----------------------------------------------------------- 2. build dataset
Write-Host "`n== 2/3 build_dataset ==" -ForegroundColor Green
& $LerobotPython "training/build_dataset.py" `
    "--raw-dir" $RawDir `
    "--root"    $DatasetRoot `
    "--repo-id" $RepoId `
    "--task"    $Task `
    "--fps"     10 `
    "--overwrite"
if ($LASTEXITCODE -ne 0) { throw "build_dataset.py failed ($LASTEXITCODE)" }

# ------------------------------------------------------------------ 3. train
# NOTE: --policy.use_amp / --policy.scheduler_decay_steps / --dataset.image_transforms.enable
# are accepted by lerobot 0.4.x; if your build rejects one, drop it from this list.
Write-Host "`n== 3/3 lerobot-train ==" -ForegroundColor Green
& $LerobotPython -m lerobot.scripts.train `
    "--dataset.repo_id=$RepoId" `
    "--dataset.root=$DatasetRoot" `
    "--dataset.image_transforms.enable=true" `
    "--policy.type=act" `
    "--policy.device=$Device" `
    "--policy.chunk_size=30" `
    "--policy.n_action_steps=15" `
    "--policy.use_amp=false" `
    "--policy.scheduler_decay_steps=$Steps" `
    "--policy.push_to_hub=false" `
    "--output_dir=$OutputDir" `
    "--job_name=act_so101" `
    "--batch_size=8" `
    "--steps=$Steps" `
    "--save_freq=$Steps" `
    "--log_freq=100" `
    "--num_workers=2" `
    "--wandb.enable=false"
if ($LASTEXITCODE -ne 0) { throw "lerobot-train failed ($LASTEXITCODE)" }

Write-Host "`ndone. checkpoints -> $OutputDir\checkpoints\last\pretrained_model" -ForegroundColor Cyan
