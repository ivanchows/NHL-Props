@echo off
rem Paper test of v3 at PTO's best price. Task Scheduler: \NHL Props\pto, :05 and :35 (NBAProps-PTO runs on the 10s and shares port 8766).
cd /d C:\Users\ivan\NHLProps
"C:\Users\ivan\.local\bin\uv.exe" run python -m nhl_props.pto_live decide >> data\live_pto\logs\decide.log 2>&1
