"""Offline regression tests for checkpoint configuration and user isolation."""
import unittest
from urllib.parse import parse_qs, urlsplit

from app.agent.memory import checkpoint_conn_string, checkpoint_key


class MemoryConfigurationTests(unittest.TestCase):
    def test_neon_ssl_translation_preserves_credentials(self):
        source = "postgresql+asyncpg://owner:p%40ss@host/neondb?ssl=verify-full"
        result = checkpoint_conn_string(source)
        self.assertEqual(result, "postgresql://owner:p%40ss@host/neondb?sslmode=verify-full")

    def test_local_connection_needs_no_ssl_query(self):
        self.assertEqual(checkpoint_conn_string("postgresql+asyncpg://u:p@localhost:5433/db"),
                         "postgresql://u:p@localhost:5433/db")

    def test_libpq_options_preserved(self):
        result = checkpoint_conn_string("postgresql://u:p@host/db?sslmode=verify-full&channel_binding=require")
        self.assertEqual(parse_qs(urlsplit(result).query),
                         {"sslmode": ["verify-full"], "channel_binding": ["require"]})

    def test_conflicting_ssl_rejected(self):
        with self.assertRaises(ValueError):
            checkpoint_conn_string("postgresql://host/db?ssl=require&sslmode=disable")

    def test_same_user_and_thread_stable_across_restarts(self):
        self.assertEqual(checkpoint_key("user-a", "thread"), checkpoint_key("user-a", "thread"))

    def test_other_user_cannot_select_same_state(self):
        self.assertNotEqual(checkpoint_key("user-a", "thread"), checkpoint_key("user-b", "thread"))

    def test_other_conversation_is_separate(self):
        self.assertNotEqual(checkpoint_key("user-a", "thread-1"), checkpoint_key("user-a", "thread-2"))

    def test_keys_are_unambiguous_and_hide_raw_identity(self):
        self.assertNotEqual(checkpoint_key("a:b", "c"), checkpoint_key("a", "b:c"))
        self.assertNotIn("user-secret", checkpoint_key("user-secret", "thread"))

    def test_empty_identity_rejected(self):
        with self.assertRaises(ValueError):
            checkpoint_key("", "thread")


if __name__ == "__main__":
    unittest.main()
