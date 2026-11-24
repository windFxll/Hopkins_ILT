[CmdletBinding()]
param(
    [string]$Python = '',
    [string]$RfCheckpoint = '',
    [string]$ProxyCheckpoint = '',
    [string]$Device = 'cuda:0',
    [ValidateSet('dataset', 'iccad2013-m1-test3')][string]$Layout = 'iccad2013-m1-test3',
    [string]$DatasetRoot = '',
    [string]$DatasetMask = 'layout_00001',
    [ValidateNotNullOrEmpty()]
    [ValidateSet('socs-tv', 'proxy-tv', 'socs-wavelet', 'proxy-wavelet', 'socs-laplacian', 'proxy-laplacian',
        'socs-flowdps', 'proxy-flowdps', 'socs-flowlps', 'proxy-flowlps', 'socs-flowsgs', 'proxy-flowsgs',
        'socs-ilt', 'proxy-ilt')]
    [string[]]$Methods = @('socs-tv', 'proxy-tv', 'socs-wavelet', 'proxy-wavelet', 'socs-laplacian',
        'proxy-laplacian', 'socs-flowdps', 'proxy-flowdps', 'socs-flowlps', 'proxy-flowlps',
        'socs-flowsgs', 'proxy-flowsgs'),
    [ValidateRange(1, 2147483647)][int]$MaxIterations = 5000,
    [switch]$NoEarlyStop,
    [double]$TvWeight = 0.01,
    [double]$WaveletWeight = 0.01,
    [ValidateRange(1, 2147483647)][int]$WaveletLevels = 3,
    [double]$LaplacianWeight = 0.01,
    [double]$BinaryWeight = 0.001,
    [double]$AreaWeight = 0.08,
    [ValidateRange(1, 2147483647)][int]$FlowSteps = 250,
    [ValidateRange(1, 2147483647)][int]$SocsInnerSteps = 1,
    [ValidateRange(1, 2147483647)][int]$ProxyInnerSteps = 1,
    [double]$DpsScale = 1.0,
    [double]$MaxGuidanceRms = 1.0,
    [double]$FlowMinCleanWeight = 0.10,
    [ValidateRange(0, 2147483647)][int]$LpsLangevinSteps = 1,
    [ValidateRange(1, 2147483647)][int]$LpsProximalSteps = 4,
    [double]$LpsProximalLr = 0.2,
    [ValidateRange(1, 2147483647)][int]$SgsSteps = 40,
    [ValidateRange(1, 2147483647)][int]$SgsLikelihoodSteps = 8,
    [ValidateRange(1, 2147483647)][int]$SgsPriorSteps = 4,
    [ValidateRange(1, 2147483647)][int]$ReportEvery = 10,
    [ValidateRange(1, 2147483647)][int]$LogEvery = 10,
    [int]$Seed = 2026,
    [string]$OutputDir = '',
    [ValidateSet('signed', 'unit')][string]$RfEncoding = 'signed',
    [ValidateSet('feature-high', 'feature-low')][string]$RfPolarity = 'feature-high',
    [switch]$AllowProxyBasisMismatch
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$opticsRoot = $repoRoot
if (!$ProxyCheckpoint) {
    $ProxyCheckpoint = Join-Path $opticsRoot 'Hopkins_surrogate/models/k8_cnn_default.pt'
}
if (!$RfCheckpoint) {
    $RfCheckpoint = Join-Path $PSScriptRoot 'checkpoints/rf_latest.pt'
}
$Methods = @($Methods | ForEach-Object { $_.ToLowerInvariant() })
if (@($Methods | Sort-Object -Unique).Count -ne $Methods.Count) {
    throw 'Pass each method once in -Methods.'
}

if (!$Python) {
    $Python = 'python'
}
$Python = (Get-Command -Name $Python -CommandType Application).Source

$checkpoints = @($proxyCheckpoint)
if ($Methods -match '-(flowdps|flowlps|flowsgs)$') {
    $checkpoints += $RfCheckpoint
}
foreach ($checkpoint in $checkpoints) {
    if (!(Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
        throw "Checkpoint not found: $checkpoint"
    }
}

if (!$OutputDir) {
    $runId = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $OutputDir = Join-Path $PSScriptRoot "runs/benchmark-$runId"
}
$OutputDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
$logFile = "$OutputDir.log"
if ((Test-Path -LiteralPath $logFile) -or
    ((Test-Path -LiteralPath $OutputDir) -and (Get-ChildItem -LiteralPath $OutputDir -Force))) {
    throw "Run output or log already exists. Choose a fresh -OutputDir: $OutputDir"
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputDir) | Out-Null

$sourcePaths = @(
    (Join-Path $PSScriptRoot 'src')
    (Join-Path $opticsRoot 'Hopkins')
    (Join-Path $opticsRoot 'Hopkins_surrogate')
    (Join-Path $opticsRoot 'ILT')
)
$pythonArgs = @(
    '-u', '-m', 'ilt_flowdps.benchmark.cli',
    '--device', $Device, '--rf-checkpoint', $RfCheckpoint,
    '--proxy-checkpoint', $proxyCheckpoint, '--output', $OutputDir,
    '--layout', $Layout.ToLowerInvariant(),
    '--max-iterations', "$MaxIterations", '--flow-steps', "$FlowSteps",
    '--tv-weight', "$TvWeight", '--wavelet-weight', "$WaveletWeight", '--laplacian-weight', "$LaplacianWeight",
    '--wavelet-levels', "$WaveletLevels", '--binary-weight', "$BinaryWeight", '--area-weight', "$AreaWeight",
    '--socs-inner-steps', "$SocsInnerSteps", '--proxy-inner-steps', "$ProxyInnerSteps",
    '--lps-langevin-steps', "$LpsLangevinSteps", '--lps-proximal-steps', "$LpsProximalSteps",
    '--lps-proximal-lr', "$LpsProximalLr", '--sgs-steps', "$SgsSteps",
    '--sgs-likelihood-steps', "$SgsLikelihoodSteps", '--sgs-prior-steps', "$SgsPriorSteps",
    '--progress-every', "$ReportEvery", '--log-every', "$LogEvery", '--seed', "$Seed",
    '--dps-scale', "$DpsScale", '--max-guidance-rms', "$MaxGuidanceRms",
    '--flow-min-clean-weight', "$FlowMinCleanWeight",
    '--rf-encoding', $RfEncoding, '--rf-polarity', $RfPolarity
)
$pythonArgs += '--methods'
$pythonArgs += $Methods
if ($DatasetRoot) {
    $pythonArgs += @('--dataset-root', $DatasetRoot)
}
$pythonArgs += @('--dataset-mask', $DatasetMask)
if ($NoEarlyStop) {
    $pythonArgs += '--no-early-stop'
}
if ($AllowProxyBasisMismatch) {
    $pythonArgs += '--allow-proxy-basis-mismatch'
}

$previousPythonPath = $env:PYTHONPATH
$previousTqdmDisable = $env:TQDM_DISABLE
$previousBytecode = $env:PYTHONDONTWRITEBYTECODE
try {
    $env:PYTHONPATH = $sourcePaths -join [IO.Path]::PathSeparator
    $env:TQDM_DISABLE = 'True'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    "Results: $OutputDir"
    "Log:     $logFile"
    # Capture stderr alongside stdout, including tracebacks and fingerprint warnings.
    $previousErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $PSNativeCommandUseErrorActionPreference = $false
    try {
        & $Python @pythonArgs 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $logFile
        $runExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorPreference
    }
    if ($runExitCode -ne 0) {
        [Console]::Error.WriteLine("Benchmark failed (exit $runExitCode). See $logFile")
        exit $runExitCode
    }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:TQDM_DISABLE = $previousTqdmDisable
    $env:PYTHONDONTWRITEBYTECODE = $previousBytecode
}
