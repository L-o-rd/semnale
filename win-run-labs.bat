@echo off
setlocal enabledelayedexpansion

rem Default values
set "LAB_NO=01"
set "LAB_EX=0"

rem Check if first argument (lab number) is provided
if "%~1"=="" (
    set "LAB_NO=01"
) else (
    if "%~1"=="01" (
        set "LAB_NO=%~1"
    ) else if "%~1"=="02" (
        set "LAB_NO=%~1"
    ) else if "%~1"=="03" (
        set "LAB_NO=%~1"
    ) else if "%~1"=="04" (
        set "LAB_NO=%~1"
    ) else if "%~1"=="05" (
        set "LAB_NO=%~1"
    ) else if "%~1"=="06" (
        set "LAB_NO=%~1"
    ) else (
        echo Usage: %~nx0 ^<lab-no^> ^(01 - 05^)
        exit /b 0
    )
)

rem Check if second argument (exercise number) is provided
if not "%~2"=="" (
    if "%~2"=="01" (
        set "LAB_EX=%~2"
    ) else if "%~2"=="02" (
        set "LAB_EX=%~2"
    ) else if "%~2"=="03" (
        set "LAB_EX=%~2"
    ) else if "%~2"=="04" (
        set "LAB_EX=%~2"
    ) else if "%~2"=="05" (
        set "LAB_EX=%~2"
    ) else if "%~2"=="06" (
        set "LAB_EX=%~2"
    ) else if "%~2"=="07" (
        set "LAB_EX=%~2"
    ) else if "%~2"=="08" (
        set "LAB_EX=%~2"
    ) else (
        set "LAB_EX=0"
    )
)

rem Run either a specific lab exercise or all exercises in a directory
if "%LAB_EX%"=="0" (
    echo Running labs\%LAB_NO%\*.py
    for %%f in (labs\%LAB_NO%\*.py) do (
        echo   - %%f..
        python "%%f"
    )
) else (
    echo Running labs\%LAB_NO%\%LAB_EX%.py
    python labs\%LAB_NO%\%LAB_EX%.py
)

endlocal