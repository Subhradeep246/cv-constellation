param(
    [ValidateSet('audit','evaluate','predict','oracle-identify')]
    [string]$Command = 'evaluate',
    [ValidateSet('fast','thorough')]
    [string]$Profile = 'fast',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Project environment is missing. Follow README.md to install it.'
}
if (-not $Output) {
    $Output = Join-Path $PSScriptRoot "outputs\$Command-$Profile"
}
$runArguments = @('-m','constellation',$Command,'--output',$Output)
if ($Profile -eq 'thorough') { $runArguments += @('--grid-step','4','--top-k','100') }
Push-Location $PSScriptRoot
try {
    & $pythonPath @runArguments
    $runExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $runExitCode
