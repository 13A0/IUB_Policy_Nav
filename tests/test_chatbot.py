"""Offline regression checks: python -m unittest discover -s tests -v."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from coursekit import nrp_llm
from src import submit as app


class ChatbotTests(unittest.TestCase):
    def setUp(self):
        app.st.cache_resource.clear()

    def tearDown(self):
        app.st.cache_resource.clear()

    def test_recall_without_api(self):
        history = [{"role": "user", "content": "What is academic misconduct?"}]
        for question in (
            "What did I just ask?",
            "What was my first question?",
            "List my previous questions",
        ):
            with self.subTest(question=question):
                self.assertIn(
                    history[0]["content"], app.memory_reply(question, history)
                )
        self.assertIn("haven't", app.memory_reply("What did I just ask?", []))

    def test_follow_up_uses_topic_but_new_question_does_not(self):
        history = [
            {"role": "user", "content": "What is academic misconduct?"},
            {"role": "user", "content": "What did I just ask?"},
        ]
        self.assertIn(
            "academic misconduct",
            app.retrieval_question("How do I report it?", history),
        )
        self.assertEqual(
            "What are tuition fees?",
            app.retrieval_question("What are tuition fees?", history),
        )
        for question in (
            "Why do students pay tuition?",
            "Explain the parking policy.",
            "Can you explain the tuition policy?",
        ):
            with self.subTest(question=question):
                self.assertEqual(question, app.retrieval_question(question, history))

    def test_recall_variants_and_repeated_requests(self):
        history = [
            {"role": "user", "content": "What is academic misconduct?"},
            {"role": "user", "content": "How do I report it?"},
        ]
        for question in (
            "Can you remember what I asked earlier?",
            "What did I tell you before?",
            "Please remind me of my last question.",
            "What did I ask you last?",
        ):
            with self.subTest(question=question):
                self.assertIn(
                    "How do I report it?", app.memory_reply(question, history)
                )
                history.append({"role": "user", "content": question})
        self.assertIn(
            "How do I report it?",
            app.memory_reply("What was my second question?", history),
        )
        self.assertIn(
            "academic misconduct",
            app.memory_reply("What did I ask before that?", history),
        )
        self.assertIn(
            "aren't enough", app.memory_reply("What was my third question?", history)
        )
        self.assertEqual(
            "1. What is academic misconduct?\n\n2. How do I report it?",
            app.memory_reply("List my previous questions", history),
        )

    def test_follow_up_chain_keeps_topic_across_recall(self):
        history = [
            {"role": "user", "content": text}
            for text in (
                "What is academic misconduct?",
                "How do I report it?",
                "Can you remember what I asked earlier?",
                "What happens after that?",
            )
        ]
        query = app.retrieval_question("Can you explain more?", history)
        self.assertIn("academic misconduct", query)
        self.assertNotIn("remember", query)
        history.append({"role": "user", "content": "What are tuition fees?"})
        query = app.retrieval_question("How do I pay them?", history)
        self.assertIn("tuition", query)
        self.assertNotIn("misconduct", query)

    def test_ui_memory_beyond_two_turns_and_session_isolation(self):
        history = [
            {"role": "user", "content": "I am a graduate student at Bloomington."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "What is academic misconduct?"},
            {
                "role": "assistant",
                "content": "Policy answer [1]",
                "sources": ["Evidence"],
            },
            {"role": "user", "content": "How do I report it?"},
            {
                "role": "assistant",
                "content": "Report answer [1]",
                "sources": ["Evidence"],
            },
        ]
        with (
            patch("coursekit.embed", return_value=[[1.0, 0.0]]) as embed,
            patch("coursekit.chat", side_effect=["Draft", "Final [1]"]) as chat,
            patch("chromadb.PersistentClient") as store,
        ):
            collection = store.return_value.get_or_create_collection.return_value
            collection.count.return_value = len(app.DOCUMENTS)
            collection.query.return_value = {"documents": [["Evidence"]]}
            ui = AppTest.from_file(
                str(app.ROOT / "src/submit.py"), default_timeout=20
            ).run()
            ui.session_state["messages"] = history.copy()
            ui.run()
            ui.chat_input[0].set_value("Can you remember what I asked earlier?").run()
            self.assertIn(
                "How do I report it?", ui.session_state["messages"][-1]["content"]
            )
            embed.assert_not_called()
            chat.assert_not_called()
            ui.chat_input[0].set_value("What happens after that?").run()
            self.assertFalse(ui.exception)
            self.assertIn("academic misconduct", embed.call_args.args[0])
            self.assertEqual(2, chat.call_count)
            for call in chat.call_args_list:
                self.assertIn("I am a graduate student at Bloomington.", call.args[0])
            other = AppTest.from_file(
                str(app.ROOT / "src/submit.py"), default_timeout=20
            ).run()
            other.chat_input[0].set_value("What was my first question?").run()
            self.assertIn("haven't", other.session_state["messages"][-1]["content"])

    @patch.object(app, "chat")
    def test_empty_evidence_does_not_call_model(self, chat):
        self.assertEqual(app.REFUSAL, app.answer("Question", [], []))
        chat.assert_not_called()

    @patch.object(app, "chat", side_effect=["Unchecked draft", "Supported answer [1]"])
    def test_only_one_review_and_same_evidence(self, chat):
        self.assertEqual(
            "Supported answer [1]", app.answer("Question", ["Policy evidence"], [])
        )
        self.assertEqual(2, chat.call_count)
        for call in chat.call_args_list:
            self.assertIn("[1] Policy evidence", call.args[0])
        self.assertEqual(0, chat.call_args.kwargs["temperature"])

    def test_empty_model_outputs_fail_closed(self):
        for outputs in ([" "], ["Draft", ""], ["Draft", RuntimeError("offline")]):
            with (
                self.subTest(outputs=outputs),
                patch.object(app, "chat", side_effect=outputs),
                self.assertRaises((ValueError, RuntimeError)),
            ):
                app.answer("Question", ["Evidence"], [])

    def test_index_refresh_resume_and_small_corpus(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(app, "ROOT", Path(directory)),
            patch.object(app, "INDEX_BATCH_SIZE", 1),
            patch.object(app, "embed") as embed,
        ):
            app.build_index.clear()
            embed.side_effect = [[[1.0, 0.0]], RuntimeError("offline")]
            with self.assertRaises(RuntimeError):
                app.build_index(("Old policy", "Second policy"))
            embed.side_effect = None
            embed.return_value = [[0.0, 1.0]]
            collection = app.build_index(("Old policy", "Second policy"))
            embed.assert_called_with(["Second policy"])
            self.assertEqual(2, collection.count())
            app.build_index.clear()
            embed.reset_mock()
            app.build_index(("Old policy", "Second policy"))
            embed.assert_not_called()
            changed = app.build_index(("New policy",))
            self.assertNotEqual(collection.name, changed.name)
            self.assertEqual(["New policy"], app.retrieve(changed, "Question"))
            empty = app.build_index(())
            self.assertEqual([], app.retrieve(empty, "Question"))
            app.build_index.clear()
            # Close Chroma's file handles before Windows removes the temporary folder.
            from chromadb.api.client import SharedSystemClient

            for key, system in list(SharedSystemClient._identifier_to_system.items()):
                if str(Path(directory) / ".chroma") == key:
                    system.stop()
                    del SharedSystemClient._identifier_to_system[key]

    def test_ui_sources_survive_rerun_and_recall_avoids_api(self):
        with (
            patch("coursekit.embed", return_value=[[1.0, 0.0]]),
            patch("coursekit.chat", side_effect=["Draft", "Final [1]"]),
            patch("chromadb.PersistentClient") as store,
        ):
            collection = store.return_value.get_or_create_collection.return_value
            collection.count.return_value = len(app.DOCUMENTS)
            collection.query.return_value = {"documents": [["Policy evidence"]]}
            ui = AppTest.from_file(
                str(app.ROOT / "src/submit.py"), default_timeout=20
            ).run()
            ui.chat_input[0].set_value("What is the policy?").run()
            self.assertFalse(ui.exception)
            self.assertEqual("Final [1]", ui.session_state["messages"][-1]["content"])
            self.assertEqual(
                ["Policy evidence"], ui.session_state["messages"][-1]["sources"]
            )
            ui.run()
            self.assertEqual(1, len(ui.expander))
            collection.query.reset_mock()
            ui.chat_input[0].set_value("What did I just ask?").run()
            self.assertIn(
                "What is the policy?", ui.session_state["messages"][-1]["content"]
            )
            collection.query.assert_not_called()
            ui.chat_input[0].set_value("   ").run()
            self.assertEqual(4, len(ui.session_state["messages"]))
            app.build_index.clear()

    def test_ui_api_failure_does_not_show_draft(self):
        with (
            patch("coursekit.embed", return_value=[[1.0, 0.0]]),
            patch(
                "coursekit.chat", side_effect=["SECRET DRAFT", RuntimeError("offline")]
            ),
            patch("chromadb.PersistentClient") as store,
        ):
            collection = store.return_value.get_or_create_collection.return_value
            collection.count.return_value = len(app.DOCUMENTS)
            collection.query.return_value = {"documents": [["Evidence"]]}
            ui = AppTest.from_file(
                str(app.ROOT / "src/submit.py"), default_timeout=20
            ).run()
            with self.assertLogs(level="ERROR"):
                ui.chat_input[0].set_value("What is the policy?").run()
            self.assertFalse(ui.exception)
            reply = ui.session_state["messages"][-1]["content"]
            self.assertIn("Please try again", reply)
            self.assertNotIn("SECRET DRAFT", reply)
            self.assertNotIn("offline", reply)
            app.build_index.clear()

    def test_ui_recall_still_works_after_retrieval_failure(self):
        with (
            patch(
                "coursekit.embed", side_effect=RuntimeError("private service detail")
            ),
            patch("coursekit.chat") as chat,
            patch("chromadb.PersistentClient") as store,
        ):
            collection = store.return_value.get_or_create_collection.return_value
            collection.count.return_value = len(app.DOCUMENTS)
            ui = AppTest.from_file(
                str(app.ROOT / "src/submit.py"), default_timeout=20
            ).run()
            with self.assertLogs(level="ERROR"):
                ui.chat_input[0].set_value("What is academic misconduct?").run()
            self.assertFalse(ui.exception)
            reply = ui.session_state["messages"][-1]["content"]
            self.assertIn("Please try again", reply)
            self.assertNotIn("private service detail", reply)
            ui.chat_input[0].set_value("What did I just ask?").run()
            self.assertIn(
                "academic misconduct", ui.session_state["messages"][-1]["content"]
            )
            chat.assert_not_called()

    def test_ui_missing_key_has_actionable_message(self):
        with (
            patch(
                "coursekit.embed", side_effect=nrp_llm.MissingKey("No API key found.")
            ),
            patch("chromadb.PersistentClient") as store,
        ):
            store.return_value.get_or_create_collection.return_value.count.return_value = len(
                app.DOCUMENTS
            )
            ui = AppTest.from_file(
                str(app.ROOT / "src/submit.py"), default_timeout=20
            ).run()
            ui.chat_input[0].set_value("What is the policy?").run()
            self.assertFalse(ui.exception)
            self.assertEqual(
                "No API key found.", ui.session_state["messages"][-1]["content"]
            )


class HelperTests(unittest.TestCase):
    @patch.object(nrp_llm, "client")
    def test_embedding_order_and_missing_results(self, client):
        api = client.return_value.__enter__.return_value
        api.embeddings.create.return_value.data = [
            SimpleNamespace(index=1, embedding=[0, 1]),
            SimpleNamespace(index=0, embedding=[1, 0]),
        ]
        self.assertEqual([[1, 0], [0, 1]], nrp_llm.embed(["first", "second"]))
        api.embeddings.create.return_value.data = [
            SimpleNamespace(index=1, embedding=[0, 1])
        ]
        with self.assertRaises(ValueError):
            nrp_llm.embed(["first", "second"])
        self.assertEqual([], nrp_llm.embed([]))

    def test_key_reload_bom_and_environment_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / ".env"
            with (
                patch.object(nrp_llm, "ENV_FILE", env),
                patch.dict(nrp_llm.os.environ, {}, clear=True),
            ):
                with self.assertRaises(nrp_llm.MissingKey):
                    nrp_llm._api_key()
                env.write_text('OPENAI_API_KEY="first"\n', encoding="utf-8-sig")
                self.assertEqual("first", nrp_llm._api_key())
                env.write_text("NRP_API_KEY=second\n", encoding="utf-8")
                self.assertEqual("second", nrp_llm._api_key())
                env.unlink()
                with patch.dict(nrp_llm.os.environ, {"NRP_API_KEY": "fallback"}):
                    self.assertEqual("fallback", nrp_llm._api_key())


if __name__ == "__main__":
    unittest.main()
