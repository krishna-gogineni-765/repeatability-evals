import json

from repeatability_evals.cli import main


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


def test_cli_offline_compare_exit_codes_and_report(tmp_path, capsys):
    same = {f"q{i}": ["backup retention is thirty days"] * 3 for i in range(10)}
    a, b = write(tmp_path, "a.json", same), write(tmp_path, "b.json", same)
    assert main(["compare", a, b, "--out", str(tmp_path / "r.md")]) == 0
    assert "no_meaningful_change" in capsys.readouterr().out

    diff = {f"q{i}": {"query": "how long?", "outputs": ["totally unrelated words here now"] * 3}
            for i in range(10)}
    c = write(tmp_path, "c.json", diff)
    assert main(["compare", a, c, "--no-text"]) == 4
    out = capsys.readouterr().out
    assert "CHANGED" in out and "totally unrelated" not in out


def test_cli_judge_function(tmp_path, capsys, monkeypatch):
    (tmp_path / "fakejudge.py").write_text(
        "def complete(prompt):\n    return '{\"score\": 5, \"reason\": \"same\"}'\n")
    monkeypatch.chdir(tmp_path)
    data = {"q": ["a", "b", "c"]}
    a, b = write(tmp_path, "a.json", data), write(tmp_path, "b.json", data)
    assert main(["compare", a, b, "--judge", "fakejudge:complete"]) == 0
