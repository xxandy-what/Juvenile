import pytest
from unittest.mock import patch, MagicMock

# 从我们的业务代码中导入要测试的函数
from ai_assistant import parse_user_intent

# ==========================================
# 前置处理：模拟 Streamlit 的 secrets
# ==========================================
# 自动化测试环境通常没有 .streamlit/secrets.toml 文件
# 这个 fixture 会在每个测试运行前，自动伪造一个假的 API Key
@pytest.fixture(autouse=True)
def mock_streamlit_secrets():
    with patch("ai_assistant.st.secrets", {"GEMINI_API_KEY": "fake_test_key_123"}):
        yield

# ==========================================
# LLM 解析意图测试 (Mocking Gemini API)
# ==========================================
class TestIntentParser:
    
    @patch("ai_assistant.genai.Client")
    def test_parse_valid_sql_intent(self, mock_client_class):
        """测试正常场景：当大模型返回完美的 JSON 时，代码能否正确解析"""
        
        # 1. 设置假装的 LLM 返回值 (Mock 注入)
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        
        mock_response = MagicMock()
        # 模拟大模型返回的一段纯净 JSON 字符串
        mock_response.text = '{"intent": "SQL_QUERY", "reasoning": "User asked for a table"}'
        mock_client_instance.models.generate_content.return_value = mock_response
        
        # 2. 执行我们的业务函数
        result = parse_user_intent("Give me a table of male deaths.")
        
        # 3. 断言结果
        assert result["intent"] == "SQL_QUERY"
        assert "table" in result["reasoning"]
        # 验证的确调用了模型一次
        mock_client_instance.models.generate_content.assert_called_once()

    @patch("ai_assistant.genai.Client")
    def test_parse_invalid_json_format(self, mock_client_class):
        """测试异常场景：当大模型“抽风”返回了非 JSON 的纯文本时，代码会不会崩溃"""
        
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        
        mock_response = MagicMock()
        # 模拟大模型忘了输出 JSON，直接输出了自然语言
        mock_response.text = "I think the user wants a SQL query, but I won't use JSON format."
        mock_client_instance.models.generate_content.return_value = mock_response
        
        # 执行
        result = parse_user_intent("Hello")
        
        # 断言结果：我们的 try-except 应该能接住 JSONDecodeError，并返回带有 ERROR intent 的安全字典
        assert result["intent"] == "ERROR"
        assert "reasoning" in result
        assert "Expecting value" in result["reasoning"] or "JSON" in result["reasoning"] # json解析报错信息

    @patch("ai_assistant.genai.Client")
    def test_api_network_error(self, mock_client_class):
        """测试异常场景：API 断网或超时"""
        
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        
        # 模拟发起网络请求时直接抛出异常
        mock_client_instance.models.generate_content.side_effect = Exception("Network timeout")
        
        result = parse_user_intent("Graph this")
        
        assert result["intent"] == "ERROR"
        assert "Network timeout" in result["reasoning"]