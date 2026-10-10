@echo off
setlocal

set "GravitySpyMetadata=.\data\gravity_spy\trainingset_v1d1_metadata.csv"
set "HarmonizedRoot=.\data\gravity_spy\trainingset_harmonized_20_512"
set "TrainingRoot=.\outputs\gravity_spy_harmonized"
set "ResearchRoot=.\outputs\research"
set "ApprovedChannels=.\outputs\gwtc\approved_channels.csv"
set "Device=cuda"

if not exist "%ResearchRoot%" mkdir "%ResearchRoot%"

echo [2/6] Treinando cinco sementes no dominio harmonizado...
python -m scripts.run_gravity_spy_seeds "%GravitySpyMetadata%" "%HarmonizedRoot%" --profile cuda_balanced --output-dir "%TrainingRoot%"
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/6] Agregando sementes...
python -m scripts.aggregate_gravity_spy "%TrainingRoot%"
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/6] Construindo referencia OOD...
python .\ondasgravitacionais_research_v9.py build-ood "%TrainingRoot%\seed_42_cuda_balanced\best_model.pt" "%GravitySpyMetadata%" "%HarmonizedRoot%" --device %Device% --output "%ResearchRoot%\ood_reference_seed42.npz"
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/6] Classificando catalogo com ensemble...
python .\ondasgravitacionais_research_v9.py classify-ensemble "%TrainingRoot%" --approved "%ApprovedChannels%" --device %Device% --ood-reference "%ResearchRoot%\ood_reference_seed42.npz" --output-dir "%ResearchRoot%\gravity_spy_ensemble"
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/6] Concluido. Consulte o relatorio final!
