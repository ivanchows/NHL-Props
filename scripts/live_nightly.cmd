@echo off
rem Paper test: new box scores, rebuild the tables, grade bets, write data\live\report.md. Task Scheduler: \NHL Props\nightly, 05:00.
cd /d C:\Users\ivan\NHLProps
"C:\Users\ivan\.local\bin\uv.exe" run python -m nhl_props.fetch recent >> data\live\logs\nightly.log 2>&1
"C:\Users\ivan\.local\bin\uv.exe" run python -m nhl_props.data >> data\live\logs\nightly.log 2>&1
"C:\Users\ivan\.local\bin\uv.exe" run python -m nhl_props.live settle >> data\live\logs\nightly.log 2>&1
