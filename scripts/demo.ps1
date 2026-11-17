# Windows + WSL: starts the lab demo (scripts/demo.sh) and keeps the WSL distribution alive;
# without an open WSL session it shuts down after a while and takes the containers with it.
#   powershell -File scripts\demo.ps1 [detector] [tracker]     |  ... -File scripts\demo.ps1 stop
param([string]$Detector = "grounding_dino", [string]$Tracker = "sam2", [string]$Distro = "Ubuntu-24.04")
$keep = Get-CimInstance Win32_Process -Filter "Name='wsl.exe'" | Where-Object { $_.CommandLine -match 'sleep infinity' }
if (-not $keep) { Start-Process wsl.exe -ArgumentList '-d', $Distro, '--', 'sleep', 'infinity' -WindowStyle Hidden }
$repo = (wsl.exe -d $Distro -- wslpath -a ($PSScriptRoot -replace '\\', '/')).Trim() -replace '/scripts$', ''
$demoArgs = if ($Detector -eq 'stop') { 'stop' } else { "$Detector $Tracker" }
wsl.exe -d $Distro -- bash -lc "cd '$repo' && scripts/demo.sh $demoArgs"
if ($Detector -ne 'stop') {
    Start-Process "http://localhost:2718"
    Start-Process "http://localhost:8081/?ds=foxglove-websocket&ds.url=ws://localhost:8766"
}
