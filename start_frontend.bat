@echo off
echo Setting up and starting the frontend server...
cd frontend

echo Installing Node.js dependencies...
call npm install

echo Starting React development server...
call npm start

pause