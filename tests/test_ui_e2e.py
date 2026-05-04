from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest

class TestAppUI:
    
    @patch("ai_assistant.genai.Client")
    def test_ai_assistant_general_chat_isolated(self, mock_client_class):
        """End-to-End UI Test using a clean wrapper to prevent context bleed."""
        
        # 1. 设置假装的 LLM 返回值
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_response = MagicMock()
        mock_response.text = '{"intent": "GENERAL_CHAT", "reasoning": "User just said hi"}'
        mock_client_instance.models.generate_content.return_value = mock_response

        # 2. 运行我们刚刚新建的“无菌”包裹器应用
        # 这里指向的是刚才新建的 test_wrapper_ai.py
        at = AppTest.from_file("test_wrapper_ai.py", default_timeout=10)
        at.secrets["GEMINI_API_KEY"] = "fake_test_key"
        at.run()
        
        # 验证启动无报错
        assert not at.exception, f"App startup error: {at.exception}"

        # 3. 模拟真实的 UI 交互
        assert len(at.chat_input) > 0, "Chat input should be present"
        at.chat_input[0].set_value("Hello AI!").run()

        # 4. 验证交互后的 UI 状态
        assert not at.exception, f"Error after submitting chat: {at.exception}"
        
        messages = at.session_state["messages"]
        assert len(messages) == 3, "History should have 3 messages"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello AI!"
        assert messages[2]["role"] == "assistant"
        assert "I am ready to help you analyze mortality data" in messages[2]["content"]