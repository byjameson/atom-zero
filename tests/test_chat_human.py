from research.chat_human import ChatHuman, extract_facts


def test_extract_facts_from_english_sentences():
    facts = dict(extract_facts(
        "My name is Alex. I live in Berlin and I like green tea."))
    assert facts["name"].startswith("Alex")
    assert facts["location"].startswith("Berlin")
    assert "green tea" in facts["likes"]


def test_chat_human_memory_tasks_and_persistence(tmp_path):
    memory = tmp_path / "memory.json"
    human = ChatHuman(memory)

    assert human.respond("my name is Alex") == "I will remember that."
    assert human.respond("add task buy milk") == "Added task 1: buy milk"
    assert "buy milk" in human.respond("list tasks")
    assert human.respond("complete task 1") == "Completed task 1."
    assert human.respond("list tasks") == "There are no open tasks."
    assert "Alex" in human.respond("what do you remember")
    assert human.respond("what is the weather?") == (
        "I do not have a language model loaded yet.")

    reloaded = ChatHuman(memory)
    assert reloaded.data["facts"][0]["value"] == "Alex"
    assert reloaded.data["tasks"][0]["status"] == "done"


def test_chat_human_task_tags_and_inventory(tmp_path):
    memory = tmp_path / "memory.json"
    human = ChatHuman(memory)

    assert human.respond("add task write report tags work,home") == (
        "Added task 1: write report (tags work, home)")
    assert human.respond("add task pay invoice tags work,finance") == (
        "Added task 2: pay invoice (tags work, finance)")
    assert human.respond("add task read book") == "Added task 3: read book"
    assert "letters, numbers" in human.respond("add task bad tags bad!")
    assert len(human.data["tasks"]) == 3
    assert human.data["tasks"][0]["tags"] == ["work", "home"]
    assert human.data["tasks"][2]["tags"] == []

    listed = human.respond("list tasks tag work")
    assert "Open tasks tagged 'work':" in listed
    assert "write report [tags: work, home]" in listed
    assert "pay invoice [tags: work, finance]" in listed
    assert "read book" not in listed
    home_only = human.respond("list tasks #home")
    assert "write report" in home_only
    assert "pay invoice" not in home_only
    assert "tagged 'home'" in home_only
    assert human.respond("list tasks tag missing") == (
        "There are no open tasks tagged 'missing'.")

    inventory = human.respond("list tags")
    assert "Task tags: 3 distinct" in inventory
    assert "2 tagged task(s) (2 open, 0 done) | 1 untagged" in inventory
    assert "- work: 2 open, 0 done, 2 total" in inventory
    assert "- finance: 1 open, 0 done, 1 total" in inventory
    assert "- home: 1 open, 0 done, 1 total" in inventory
    assert human.respond("tags") == inventory

    assert human.respond("tag task 3 work") == "Set tags on task 3: work."
    assert human.respond("tag task 3 clear") == "Cleared tags on task 3."
    assert human.respond("tag task 99 work") == (
        "I could not find that task.")
    assert "letters, numbers" in human.respond("tag task 1 bad!")
    assert human.respond("tag task 1") == (
        "Use 'tag task ID tag1,tag2' to set task tags.")

    reloaded = ChatHuman(memory)
    assert reloaded.data["tasks"][0]["tags"] == ["work", "home"]
    assert reloaded.data["tasks"][1]["tags"] == ["work", "finance"]
    assert reloaded.data["tasks"][2]["tags"] == []
    assert reloaded.respond("list tags").startswith("Task tags: 3 distinct")


def test_chat_human_confirmed_tag_rename(tmp_path):
    memory = tmp_path / "memory.json"
    human = ChatHuman(memory)
    human.respond("add task write report tags work,home")
    human.respond("add task pay invoice tags work,finance")

    preview = human.respond("rename tag work job")
    assert "Rename tag 'work' to 'job' on 2 task(s)" in preview
    assert "confirm tag rename" in preview
    assert human.pending_tag_rename == {
        "old": "work", "new": "job"}
    assert human.data["tasks"][0]["tags"] == ["work", "home"]

    assert human.respond("confirm tag rename") == (
        "Renamed tag 'work' to 'job' on 2 task(s).")
    assert human.pending_tag_rename is None
    assert human.data["tasks"][0]["tags"] == ["job", "home"]
    assert human.data["tasks"][1]["tags"] == ["job", "finance"]
    assert human.respond("confirm tag rename") == (
        "There is no pending tag rename.")

    assert human.respond("rename tag missing job") == (
        "Tag 'missing' is not used.")
    assert "must be different" in human.respond("rename tag job job")
    assert "letters, numbers" in human.respond("rename tag job bad!")

    second = human.respond("rename tag job to office")
    assert "Rename tag 'job' to 'office'" in second
    assert human.pending_tag_rename == {
        "old": "job", "new": "office"}
    assert human.respond("cancel tag rename") == "Tag rename cancelled."
    assert human.pending_tag_rename is None
    assert human.data["tasks"][0]["tags"] == ["job", "home"]
    assert human.respond("cancel tag rename") == (
        "There is no pending tag rename.")

    reloaded = ChatHuman(memory)
    assert reloaded.data["tasks"][0]["tags"] == ["job", "home"]
    assert reloaded.data["tasks"][1]["tags"] == ["job", "finance"]
