$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $env:USERPROFILE "anaconda3\envs\imageclass\python.exe"

Set-Location $projectDir

Write-Host "Starting Animal Image Classification API..."

$server = Start-Process -FilePath $python -ArgumentList "-m uvicorn main:application --host 127.0.0.1 --port 8000" -WorkingDirectory $projectDir -PassThru

Start-Sleep -Seconds 3
Start-Process "http://127.0.0.1:8000/web"

Write-Host "Project launched! Server PID: $($server.Id)"
Write-Host "Stop server with: Stop-Process -Id $($server.Id)"
