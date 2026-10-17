param(
    [string]$PythonExecutable = "$env:USERPROFILE\anaconda3\envs\gw-v7\python.exe",
    [string]$Inventory = "outputs\research\official_inventory_20260918_195429.json"
)

# Prepared for user execution. No training, GPU use or canonical-data replacement.
# Download is streamed to disk; incomplete transfers are retained for resumption.
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$PreviousPythonPath = $env:PYTHONPATH
$PreviousNoUserSite = $env:PYTHONNOUSERSITE
$TranscriptStarted = $false
Push-Location -LiteralPath $ProjectRoot
try {
    if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
        throw "Python nao encontrado: $PythonExecutable"
    }
    if (-not [System.IO.Path]::IsPathRooted($Inventory)) {
        $Inventory = Join-Path $ProjectRoot $Inventory
    }
    $InventoryObject = Get-Content -LiteralPath $Inventory -Raw | ConvertFrom-Json
    if ($InventoryObject.record_id -ne 1486046) { throw "Registro Zenodo inesperado." }
    $Candidates = @($InventoryObject.remote_files | Where-Object { $_.name -eq "trainingsetv1d1.h5" })
    if ($Candidates.Count -ne 1) { throw "O inventario deve conter exatamente um trainingsetv1d1.h5." }
    $Descriptor = $Candidates[0]
    $ExpectedBytes = [long]$Descriptor.size
    $ChecksumText = [string]$Descriptor.checksum
    if ($ExpectedBytes -ne 3051325064 -or $ChecksumText -ne "md5:a6665b4ebee7b4ea57164ee743c2bd11") {
        throw "Tamanho/checksum mudou em relacao ao inventario revisado. Nao baixar sem revisar a procedencia."
    }
    $DownloadUri = [uri]$Descriptor.url
    if ($DownloadUri.Scheme -ne "https" -or $DownloadUri.Host -ne "zenodo.org" -or
        $DownloadUri.AbsolutePath -ne "/api/records/1486046/files/trainingsetv1d1.h5/content" -or
        $DownloadUri.Query -ne "" -or $DownloadUri.UserInfo -ne "") {
        throw "URL inesperada no inventario."
    }
    $DestinationRoot = Join-Path $ProjectRoot "data\gravity_spy_official"
    New-Item -ItemType Directory -Path $DestinationRoot -Force | Out-Null
    $Target = Join-Path $DestinationRoot "trainingsetv1d1.h5"
    $Partial = Join-Path $DestinationRoot "trainingsetv1d1.h5.partial"
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $Suffix = [guid]::NewGuid().ToString("N").Substring(0, 8)
    $RunName = "official_hdf5_${Stamp}_${Suffix}"
    $ResearchRoot = Join-Path $ProjectRoot "outputs\research"
    $LogRoot = Join-Path $ResearchRoot "logs"
    New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
    $LogPath = Join-Path $LogRoot "$RunName.log"
    $SchemaPath = Join-Path $ResearchRoot "$RunName.schema.json"
    Start-Transcript -Path $LogPath | Out-Null
    $TranscriptStarted = $true
    Write-Host "Log: $LogPath"
    Write-Host "Arquivo oficial: $Target (3.051.325.064 bytes)"

    if (-not (Test-Path -LiteralPath $Target)) {
        $ExistingBytes = 0L
        if (Test-Path -LiteralPath $Partial) { $ExistingBytes = (Get-Item -LiteralPath $Partial).Length }
        if ($ExistingBytes -gt $ExpectedBytes) {
            throw "Arquivo parcial maior que o esperado. Ele foi preservado; envie o log."
        }
        $DriveRoot = [System.IO.Path]::GetPathRoot($DestinationRoot)
        $DriveInfo = New-Object System.IO.DriveInfo -ArgumentList $DriveRoot
        if ($DriveInfo.AvailableFreeSpace -lt ($ExpectedBytes - $ExistingBytes + 2GB)) {
            throw "Espaco insuficiente: exige o restante do download mais 2 GiB de reserva."
        }
        if ($ExistingBytes -lt $ExpectedBytes) {
            # Conda and Windows can both provide curl.exe. Invoke one scalar
            # path, never the Source properties of a collection of commands.
            $CurlCommand = Get-Command curl.exe -CommandType Application -ErrorAction Stop |
                Select-Object -First 1
            $CurlExecutable = [string]$CurlCommand.Source
            if ([string]::IsNullOrWhiteSpace($CurlExecutable) -or
                -not (Test-Path -LiteralPath $CurlExecutable -PathType Leaf)) {
                throw "Nao foi possivel selecionar um executavel curl.exe valido."
            }
            Write-Host "Executavel de download: $CurlExecutable"
            Write-Host "[1/3] Baixando/retomando somente o HDF5; nenhum TAR sera baixado."
            & $CurlExecutable --fail --location --connect-timeout 30 --retry 3 --continue-at - `
                --output $Partial $DownloadUri.AbsoluteUri
            if ($LASTEXITCODE -ne 0) {
                throw "Download interrompido. O parcial foi preservado. Repita este roteiro; se persistir, envie o log."
            }
        }
        $FileToVerify = $Partial
    } else {
        Write-Host "[1/3] Arquivo existente: sera verificado, sem sobrescrita ou novo download."
        $FileToVerify = $Target
    }

    Write-Host "[2/3] Conferindo tamanho e MD5 oficial; a leitura e sequencial em disco."
    if ((Get-Item -LiteralPath $FileToVerify).Length -ne $ExpectedBytes) {
        throw "Tamanho incorreto. Arquivo preservado; a inspecao nao sera iniciada."
    }
    $ActualMd5 = (Get-FileHash -LiteralPath $FileToVerify -Algorithm MD5).Hash.ToLowerInvariant()
    if ($ActualMd5 -ne "a6665b4ebee7b4ea57164ee743c2bd11") {
        throw "MD5 incorreto. Arquivo preservado; envie o log antes de repetir."
    }
    if ($FileToVerify -eq $Partial) {
        # No -Force: an existing destination is never replaced.
        Move-Item -LiteralPath $Partial -Destination $Target
    }
    Write-Host "Integridade confirmada: $ActualMd5"
    $env:PYTHONPATH = Join-Path $ProjectRoot "src"
    $env:PYTHONNOUSERSITE = "1"
    Write-Host "[3/3] Inspecionando somente nomes, formas e tipos; sem ler valores dos arrays."
    & $PythonExecutable -m gwproject.research_cli inspect-official `
        --local-hdf5 $Target --output $SchemaPath
    if ($LASTEXITCODE -ne 0) { throw "Inspecao falhou. HDF5 preservado; envie o log." }
    Write-Host "Concluido. Envie o JSON: $SchemaPath"
    Write-Host "E o log: $LogPath"
    Write-Host "Nao envie o HDF5 de 3 GB. Ainda nao foi estabelecido pareamento entre os dados."
} finally {
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
    $env:PYTHONPATH = $PreviousPythonPath
    $env:PYTHONNOUSERSITE = $PreviousNoUserSite
    Pop-Location
}
