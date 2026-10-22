from types import SimpleNamespace

from app import ask as ask_module


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=34,
        ),
        text="回答",
    )

    class FakeChat:
        def send_message(self, question):
            assert question == "質問"
            return response

    assert ask_module.ask(FakeChat(), "質問") == "回答"
    output = capsys.readouterr().out
    assert "[応答時間]" in output
    assert "[トークン] 入力: 12, 出力: 34\n" in output


def test_main_keeps_chat_for_follow_up_questions(monkeypatch, capsys):
    questions_by_model = {model: [] for model in ask_module.MODELS}
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=34,
        ),
        text="回答",
    )

    class FakeChat:
        def __init__(self, model):
            self.model = model

        def send_message(self, question):
            questions_by_model[self.model].append(question)
            return response

    class FakeChats:
        def create(self, *, model):
            assert model in ask_module.MODELS
            return FakeChat(model)

    class FakeClient:
        def __init__(self, api_key):
            assert api_key == "test-key"
            self.chats = FakeChats()

    answers = iter(["補足して", "exit"])
    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr("sys.argv", ["app.ask", "LoRA とは何ですか？"])
    ask_module.main()

    assert questions_by_model == {
        model: [
            "LoRA とは何ですか？",
            "LoRA とは何ですか？",
            "LoRA とは何ですか？",
            "補足して",
            "補足して",
            "補足して",
        ]
        for model in ask_module.MODELS
    }
    output = capsys.readouterr().out
    assert output.count("--- model=") == 12
    assert output.count("[応答時間]") == 12
    assert output.count("[トークン]") == 12
    assert output.count("回答") == 12