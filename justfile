save message="update":
    git add -A
    git commit -m "{{message}}" || true
    git push

open:
    code .

deps:
    test -d venv || (command -v python3 >/dev/null 2>&1 && python3 -m venv venv || python -m venv venv)
    if [ -f venv/bin/pip ]; then venv/bin/pip install -r requirements.txt; else venv/Scripts/pip.exe install -r requirements.txt; fi
