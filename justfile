save message="update":
    git add -A
    git commit -m "{{message}}" || true
    git push

open:
    code .

deps:
    test -d venv || python3 -m venv venv
    venv/bin/pip install -r requirements.txt
