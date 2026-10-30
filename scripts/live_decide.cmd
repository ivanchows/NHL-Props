@echo off
rem Paper test: log Novig SOG quotes and decide each game at two stages. Task Scheduler: \NHL Props\decide, every 30 min.
cd /d C:\Users\ivan\NHLProps
"C:\Users\ivan\.local\bin\uv.exe" run python -m nhl_props.live decide >> data\live\logs\decide.log 2>&1
