save message="update":
    git add -A
    git commit -m "{{message}}" || true
    git push

open:
    code .

deps:
    test -d venv || python -m venv venv
    venv/Scripts/pip install -r requirements.txt
