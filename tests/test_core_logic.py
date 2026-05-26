import pytest
import pandas as pd
import numpy as np
from utils import safe_div, sql_ident
from ai_assistant import execute_read_only_sql

# ==========================================
# 1. appv2.py 核心工具函数测试
# ==========================================
def test_safe_div_normal_and_zero():
    """测试安全除法：正常计算与除零保护"""
    n = pd.Series([10.0, 20.0, 5.0])
    d = pd.Series([2.0, 0.0, np.nan])  # 包含 0 和 NaN
    
    result = safe_div(n, d)
    
    assert result[0] == 5.0
    assert pd.isna(result[1]), "除以 0 应该返回 NaN 而不是 Inf"
    assert pd.isna(result[2]), "除以 NaN 应该返回 NaN"

def test_sql_ident():
    """测试 SQL 标识符转义 (防止简单的注入或语法错误)"""
    assert sql_ident("Death_Count") == '"Death_Count"'
    assert sql_ident('User"Name') == '"User""Name"', "双引号应该被正确转义"

# ==========================================
# 2. ai_assistant.py 安全防火墙测试
# ==========================================
class TestSQLFirewall:
    """测试 LLM SQL 执行器的恶意关键字拦截机制"""
    
    def test_block_non_select_format(self):
        """测试第一层防御：拒绝非 SELECT/WITH 开头的语句"""
        malicious_sql_1 = "DROP TABLE current_working_set;"
        malicious_sql_2 = "INSERT INTO current_working_set VALUES (1, 2, 3);"
        
        with pytest.raises(ValueError, match="Invalid Query Format"):
            execute_read_only_sql(malicious_sql_1, "dummy.parquet", False)
            
        with pytest.raises(ValueError, match="Invalid Query Format"):
            execute_read_only_sql(malicious_sql_2, "dummy.parquet", False)
            
    def test_block_file_io_functions(self):
        """测试第二层防御：拦截危险关键字（绕过了第一层的前提下）"""
        # 使用精确匹配的黑名单词 READ_CSV 和 SYSTEM
        malicious_sql_1 = "SELECT * FROM read_csv('/etc/passwd');"
        malicious_sql_2 = "SELECT SYSTEM('rm -rf /');"
        
        with pytest.raises(ValueError, match="Security Restriction"):
            execute_read_only_sql(malicious_sql_1, "dummy.parquet", False)
            
        with pytest.raises(ValueError, match="Security Restriction"):
            execute_read_only_sql(malicious_sql_2, "dummy.parquet", False)

    def test_allow_valid_select(self):
        """测试合法 SQL 可以通过安全检查到达执行层"""
        valid_sql = "SELECT SUM(Death_Count) FROM current_working_set"
        try:
            execute_read_only_sql(valid_sql, "non_existent.parquet", False)
        except ValueError as e:
            # 只要不是防火墙的安全报错就行（说明它被放行去尝试读数据了）
            assert "Security Restriction" not in str(e)
            assert "Invalid Query Format" not in str(e)