@echo off
echo Building all presentations...
echo.
call npm run build
echo.
echo Generating index page...
call npm run build:index
echo.
echo Done! Check the public/ folder for output.
pause
