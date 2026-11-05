@echo off
rem usage: decrypt.cmd <password> [archive]  -> extracts .\hlc-samples\ (openssl from Git for Windows, tar from Windows 10+)
if "%~1"=="" (echo usage: decrypt.cmd ^<password^> [archive] & exit /b 1)
set "A=%~2"
if "%A%"=="" set "A=%~dp0hlc-samples.tar.xz.enc"
set "OSSL=openssl"
where openssl >nul 2>nul || set "OSSL=%ProgramFiles%\Git\usr\bin\openssl.exe"
"%OSSL%" enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass "pass:%~1" -in "%A%" -out "%TEMP%\hlc-samples.tar.xz" || exit /b 1
tar xJf "%TEMP%\hlc-samples.tar.xz" || exit /b 1
del "%TEMP%\hlc-samples.tar.xz"
echo ok: .\hlc-samples
