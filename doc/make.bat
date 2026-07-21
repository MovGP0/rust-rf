@ECHO OFF
SETLOCAL
PUSHD %~dp0
IF "%1"=="" GOTO help
python -m sphinx -W --keep-going -M %1 source build
GOTO end
:help
python -m sphinx -M help source build
:end
POPD
ENDLOCAL
