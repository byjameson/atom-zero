from atomzero.quickstart import reset_demo


def test_reset_demo_removes_db(tmp_path):
    db = tmp_path / "demo.db"
    db.write_text("old", encoding="utf-8")
    reset_demo(db)
    assert not db.exists()
    reset_demo(db)  # missing is fine


def test_reset_demo_missing_is_noop(tmp_path):
    reset_demo(tmp_path / "missing.db")
    assert not (tmp_path / "missing.db").exists()
