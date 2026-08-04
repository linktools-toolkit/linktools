import pytest

from linktools.ai.acp.sessions import validate_session_paths


def test_session_paths_allow_project_children_and_reject_escape(tmp_path) -> None:
    project = tmp_path / "project"
    child = project / "child"
    outside = tmp_path / "outside"
    child.mkdir(parents=True)
    outside.mkdir()

    cwd, additional = validate_session_paths(
        project_root=project,
        cwd=str(child),
        additional_directories=[str(child), str(child)],
    )
    assert cwd == str(child.resolve())
    assert additional == (str(child.resolve()),)

    with pytest.raises(Exception):
        validate_session_paths(project_root=project, cwd=str(outside), additional_directories=[])
