@echo off
rem Paper test of v3: tonight's lineups + the TOI model. Task Scheduler: \NHL Props\decide3, every 30 min.
cd /d C:\Users\ivan\NHLProps
"C:\Users\ivan\.local\bin\uv.exe" run python -m nhl_props.live_v3 decide >> data\live_v3\logs\decide.log 2>&1
