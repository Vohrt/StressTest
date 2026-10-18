# Builds CpuStressTest (WinUI app + measurement engine + module self-tests).
#
#   .\build.ps1                     # Release build of the whole solution
#   .\build.ps1 -Configuration Debug
#   .\build.ps1 -SelfTests          # only the module self-tests
#
# Artifacts (Release):
#   build\cst.exe              the app - double-click opens the GUI, arguments run the CLI
#   build\selftest_*.exe       module self-tests
#   build\fractal_dump.exe     AIDA64 frame alignment dumper
#   build\lib\cst_engine.lib   the measurement engine, statically linked into cst.exe
#
# Debug builds go to build\debug\ (and build\lib\debug\) so they can never
# overwrite the Release binaries the recorded reference fingerprints were made
# with - a Debug build may produce different low-order floating point bits.
param(
    [ValidateSet('Release', 'Debug')]
    [string]$Configuration = 'Release',
    [switch]$SelfTests
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

function Find-MSBuild {
    $patterns = @(
        (Join-Path $env:ProgramFiles 'Microsoft Visual Studio\*\*\MSBuild\Current\Bin\MSBuild.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\*\*\MSBuild\Current\Bin\MSBuild.exe')
    )
    foreach ($pattern in $patterns) {
        $hit = Get-ChildItem $pattern -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    throw 'MSBuild.exe not found - install the Visual Studio C++ workload'
}

$msbuild = Find-MSBuild
$projects = if ($SelfTests) {
    @('vs\selftest_crypto.vcxproj', 'vs\selftest_image.vcxproj', 'vs\selftest_memory.vcxproj')
} else {
    @('CpuStressTest.sln')
}

foreach ($project in $projects) {
    Write-Host "== building $project ($Configuration|x64)"
    & $msbuild (Join-Path $root $project) /restore /m /nologo /v:m /p:Configuration=$Configuration /p:Platform=x64
    if ($LASTEXITCODE -ne 0) { throw "$project build failed ($LASTEXITCODE)" }
}

Write-Host ''
Write-Host 'Done. Artifacts:'
$out = if ($Configuration -eq 'Release') { Join-Path $root 'build' } else { Join-Path $root 'build\debug' }
Get-ChildItem (Join-Path $out '*.exe') -ErrorAction SilentlyContinue |
    ForEach-Object { "  $($_.FullName.Substring($root.Length + 1))" }
