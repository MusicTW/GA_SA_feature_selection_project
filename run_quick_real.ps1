$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
chcp 65001 > $null
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Set-Location -LiteralPath $PSScriptRoot
$DataPath = Join-Path $PSScriptRoot "data\raw\train_transaction.csv"
python -u run_experiment.py --mode real --data-path $DataPath --preset quick --device cpu
