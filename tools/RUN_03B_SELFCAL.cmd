@echo off
REM 双击本文件即可运行「自校准数据量扫描」（独立于 Codex 会话，不会被回合中断杀掉）
cd /d "%~dp0..\src"
python exp_v3_03b_selfcal_sweep.py
echo.
echo finished; see ..\outputs\exp_v3_03b_selfcal_sweep.csv
pause
