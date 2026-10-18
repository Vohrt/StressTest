# UI Automation smoke test for the CpuStressTest GUI.
#
#   powershell -ExecutionPolicy Bypass -File tools\gui_smoke.ps1
#
# Launches build\cst.exe (no arguments -> GUI) and drives the parts of the UI
# that are easy to get wrong:
#
#   1. clearing the selection and running -> the GUI must refuse with a hint
#   2. selecting a single workload (CPU CheckMate) and running it -> the run
#      must complete, verify against the recorded reference and update the row
#   3. selecting everything, starting a sustained run and pressing stop -> the
#      run must stop
#   4. the "记录为参考…" button -> a confirmation dialog must appear and 取消
#      must not start anything
#   5. 运行并记录 must run and re-record the reference (reference.txt is backed
#      up and restored around the test)
#
# Exit code 0 = all scenarios behaved, 1 = something is wrong.  The script
# writes a short temporary settings file first so the runs take seconds.
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root 'build\cst.exe'
if (-not (Test-Path $exe)) { throw "build\cst.exe not found - build the solution first" }

Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes

function Find-ById($root, $id) {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::AutomationIdProperty, $id)
    return $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $cond)
}

function Text-Of($element) {
    if (-not $element) { return '' }
    return $element.Current.Name
}

function Invoke-Button($window, $id) {
    $button = Find-ById $window $id
    if (-not $button) { throw "$id not found" }
    $pattern = $button.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
    $pattern.Invoke()
}

function Find-ByName($root, $name) {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty, $name)
    return $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $cond)
}

function Invoke-Element($element) {
    $pattern = $element.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
    $pattern.Invoke()
}

# ContentDialog buttons can show up inside the window's automation tree or off
# it; look in both places, and wait a moment for the dialog animation.
function Wait-ForDialogButton($window, $root, $name, [int]$seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    do {
        $found = Find-ByName $window $name
        if (-not $found) { $found = Find-ByName $root $name }
        if ($found) { return $found }
        Start-Sleep -Milliseconds 300
    } while ((Get-Date) -lt $deadline)
    return $null
}

function Set-CheckBox($window, $id, [bool]$value) {
    $element = Find-ById $window $id
    if (-not $element) { throw "$id not found" }
    # CheckBox exposes TogglePattern, RadioButton exposes SelectionItemPattern.
    $pattern = $null
    try { $pattern = $element.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern) } catch { $pattern = $null }
    if ($pattern) {
        $on = ($pattern.Current.ToggleState -eq [System.Windows.Automation.ToggleState]::On)
        if ($on -ne $value) { $pattern.Toggle() }
        return
    }
    $select = $element.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern)
    if ($value) { $select.Select() }
}

# Wait until the status line reports that a run ended; returns the final text.
function Wait-Finished($window, $process, [int]$seconds) {
    $status = Find-ById $window 'StatusText'
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        if ($process.HasExited) { throw "GUI exited while running (code $($process.ExitCode))" }
        $text = Text-Of $status
        if ($text.StartsWith('完成') -or $text.StartsWith('已停止') -or $text.StartsWith('结束') -or
            $text.StartsWith('运行失败')) {
            return $text
        }
    }
    throw "run did not finish within $seconds s (status: $(Text-Of $status))"
}

# --- short settings for the test run (restored afterwards) -------------------
$settingsDir = Join-Path $env:LOCALAPPDATA 'CpuStressTestGui'
New-Item -ItemType Directory -Force -Path $settingsDir | Out-Null
$settingsFile = Join-Path $settingsDir 'settings.ini'
$backupFile = "$settingsFile.smoketest-backup"
$hadSettings = Test-Path $settingsFile
if ($hadSettings) { Copy-Item $settingsFile $backupFile -Force }
@(
    'seconds=2'
    'minutes=1'
    'threads=0'
    'sustained=0'
    'csv=0'
    'csv_path=results.csv'
    'all_selected=1'
    'window_width=1240'
    'window_height=860'
) | Set-Content -Path $settingsFile -Encoding ASCII

# The record-reference scenario rewrites reference.txt; put the user's copy back
# afterwards no matter how the test ends.
$referenceFile = Join-Path $root 'reference.txt'
$referenceBackup = "$referenceFile.smoketest-backup"
if (Test-Path $referenceFile) { Copy-Item $referenceFile $referenceBackup -Force }

$process = Start-Process -FilePath $exe -PassThru
try {
    Start-Sleep -Seconds 8
    if ($process.HasExited) { throw "GUI exited during startup (code $($process.ExitCode))" }

    $rootElement = [System.Windows.Automation.AutomationElement]::RootElement
    $windowCondition = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty, 'CpuStressTest')
    $window = $rootElement.FindFirst([System.Windows.Automation.TreeScope]::Children, $windowCondition)
    if (-not $window) { throw 'main window not found through UI Automation' }
    Write-Host '[1/6] window found'

    # 1. nothing selected must not start a run ---------------------------------
    Invoke-Button $window 'SelectNoneButton'
    Start-Sleep -Milliseconds 500
    Invoke-Button $window 'RunButton'
    Start-Sleep -Seconds 1
    $status = Text-Of (Find-ById $window 'StatusText')
    if ($status -notlike '*没有选中*') { throw "expected the 'nothing selected' hint, got: $status" }
    Write-Host "[2/6] empty selection handled: $status"

    # 2. run exactly one workload ---------------------------------------------
    Set-CheckBox $window 'Select_checkmate' $true
    Invoke-Button $window 'RunButton'
    $final = Wait-Finished $window $process 120
    $row = Text-Of (Find-ById $window 'Status_checkmate')
    $value = Text-Of (Find-ById $window 'Value_checkmate')
    Write-Host "[3/6] single run: $final | row: '$row' value: '$value'"
    if ($final -notlike '完成*所有测试通过校验*') { throw "single run did not pass verification: $final" }
    if ($row -notlike '*校验通过*') { throw "checkmate row did not report verification: $row" }
    if ([string]::IsNullOrWhiteSpace($value) -or $value -eq '—') { throw 'checkmate row has no throughput value' }

    # 3. sustained run + stop --------------------------------------------------
    Invoke-Button $window 'SelectAllButton'
    Set-CheckBox $window 'SustainedRadio' $true
    Invoke-Button $window 'RunButton'
    Start-Sleep -Seconds 5
    Invoke-Button $window 'StopButton'
    $stopped = Wait-Finished $window $process 90
    Write-Host "[4/6] stop scenario: $stopped"
    if ($stopped -notlike '已停止*') { throw "stop did not report a cancelled run: $stopped" }

    # 4. record-reference dialog: 取消 must do nothing -------------------------
    Invoke-Button $window 'SelectNoneButton'
    Set-CheckBox $window 'Select_checkmate' $true
    Set-CheckBox $window 'SingleRadio' $true
    Invoke-Button $window 'RecordRefButton'
    $cancel = Wait-ForDialogButton $window $rootElement '取消' 8
    if (-not $cancel) { throw 'record-reference dialog did not open' }
    Invoke-Element $cancel
    Start-Sleep -Seconds 1
    $status = Text-Of (Find-ById $window 'StatusText')
    if ($status -like '正在运行*') { throw "cancelling the dialog started a run anyway: $status" }
    Write-Host '[5/6] record-reference dialog cancels cleanly'

    # 5. 运行并记录 must run and rewrite the reference -------------------------
    Invoke-Button $window 'RecordRefButton'
    $go = Wait-ForDialogButton $window $rootElement '运行并记录' 8
    if (-not $go) { throw 'record-reference dialog did not reopen' }
    Invoke-Element $go
    $final = Wait-Finished $window $process 180
    Write-Host "[6/6] record-reference run: $final"
    if ($final -notlike '完成*') { throw "record run did not finish cleanly: $final" }

    Write-Host 'GUI SMOKE TEST PASSED'
    exit 0
}
finally {
    if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
    # Put the user's own settings and references back; the test must not leave
    # seconds=2 in place or a re-recorded fingerprint behind.
    if ($hadSettings) { Copy-Item $backupFile $settingsFile -Force }
    else { Remove-Item $settingsFile -Force -ErrorAction SilentlyContinue }
    Remove-Item $backupFile -Force -ErrorAction SilentlyContinue
    if (Test-Path $referenceBackup) { Copy-Item $referenceBackup $referenceFile -Force }
    Remove-Item $referenceBackup -Force -ErrorAction SilentlyContinue
}
