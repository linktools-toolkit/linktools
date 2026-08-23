from pathlib import Path

path = Path("tests/ai/test_asset_repository.py")
text = path.read_text()
old = '''    with pytest.raises(ValueError):
        SkillSpec("foo", True, content)
'''
new = '''    with pytest.raises(TypeError):
        SkillSpec("foo", True, content)
'''
count = text.count(old)
if count != 1:
    raise SystemExit(f"skill revision type test: expected one match, found {count}")
path.write_text(text.replace(old, new))
