"""
Lab #4 Autograder — kiểm thử tools.py và template.py.

Chạy từ thư mục gốc: python -m pytest autograder/test_agent.py -v
"""

import sys
import os
import json
import shutil
from pathlib import Path
import pytest

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "solution-code"))
STARTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "starter-code"))

for code_dir in [SOLUTION_DIR, STARTER_DIR]:
    if os.path.exists(code_dir) and code_dir not in sys.path:
        sys.path.insert(0, code_dir)

try:
    from template import ChatbotBaseline, ToolCallingAgent
except ModuleNotFoundError:
    from agent import ChatbotBaseline, ToolCallingAgent

from tools import search_product_catalog, submit_support_ticket, TOOL_MAP, TOOL_DEFINITIONS
import tools


@pytest.fixture(autouse=True)
def isolated_raw_data(tmp_path, monkeypatch):
    """Mỗi test dùng bản sao riêng, không thêm ticket vào dữ liệu mẫu."""
    source = Path(__file__).resolve().parents[1] / "raw-data"
    for filename in ("product_catalog.json", "support_tickets.json"):
        shutil.copyfile(source / filename, tmp_path / filename)
    monkeypatch.setattr(tools, "RAW_DATA_DIR", str(tmp_path))
    return tmp_path


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: search_product_catalog — xe điện giá dưới 600 triệu
# ═══════════════════════════════════════════════════════════════════════════

def test_catalog_search_xe_dien_under_600m():
    results = search_product_catalog(category="xe_dien", max_price=600000000)
    assert isinstance(results, list)
    assert len(results) == 2  # VF 3 (315M) + VF 5 Plus (548M)
    names = [p["name"] for p in results]
    assert "VinFast VF 3" in names
    assert "VinFast VF 5 Plus" in names


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: search_product_catalog — du lịch giá dưới 6 triệu
# ═══════════════════════════════════════════════════════════════════════════

def test_catalog_search_du_lich_under_6m():
    results = search_product_catalog(category="du_lich", max_price=6000000)
    assert isinstance(results, list)
    assert len(results) == 2  # Nha Trang (3.5M) + Phú Quốc (5.2M)
    names = [p["name"] for p in results]
    assert "Vinpearl Resort & Spa Nha Trang" in names
    assert "Vinpearl Discovery Phú Quốc" in names


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: submit_support_ticket — tạo ticket thành công
# ═══════════════════════════════════════════════════════════════════════════

def test_submit_ticket_creates_ticket():
    result = submit_support_ticket(
        customer_name="Test User",
        issue_description="Test issue for autograder",
        priority="high"
    )
    assert isinstance(result, dict)
    assert "ticket_id" in result
    assert result["ticket_id"].startswith("TK-")
    assert result["customer_name"] == "Test User"
    assert result["priority"] == "high"
    assert result["status"] == "open"


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: ChatbotBaseline — không gọi tool
# ═══════════════════════════════════════════════════════════════════════════

def test_chatbot_baseline_no_tools():
    chatbot = ChatbotBaseline()
    res = chatbot.query("Tôi muốn xem xe VinFast")
    assert res["status"] == "success"
    assert len(res["tool_calls"]) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: ToolCallingAgent — single tool (catalog search)
# ═══════════════════════════════════════════════════════════════════════════

def test_agent_single_tool_catalog():
    agent = ToolCallingAgent(max_iterations=5)
    res = agent.run("Tôi muốn xem xe điện VinFast giá dưới 600 triệu.")
    assert res["status"] == "completed"
    assert res["iterations"] == 1
    assert "VF 3" in res["answer"] or "VF 5" in res["answer"]


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: ToolCallingAgent — single tool (ticket submit)
# ═══════════════════════════════════════════════════════════════════════════

def test_agent_single_tool_ticket():
    agent = ToolCallingAgent(max_iterations=5)
    res = agent.run("Tôi tên Lê Minh Khoa, xe VF 8 của tôi bị lỗi hệ thống ADAS. Đây là vấn đề nghiêm trọng, cần xử lý gấp.")
    assert res["status"] == "completed"
    assert res["iterations"] == 1
    assert "TK-" in res["answer"]
    assert "Lê Minh Khoa" in res["answer"]


# ═══════════════════════════════════════════════════════════════════════════
# Test 7: ToolCallingAgent — FAQ (no tool call)
# ═══════════════════════════════════════════════════════════════════════════

def test_agent_faq_no_tool():
    agent = ToolCallingAgent(max_iterations=5)
    res = agent.run("Chính sách bảo hành pin xe điện VinFast kéo dài bao lâu?")
    assert res["status"] == "completed"
    assert res["iterations"] == 1
    assert "bảo hành" in res["answer"].lower() or "10 năm" in res["answer"]


# ═══════════════════════════════════════════════════════════════════════════
# Test 8: ToolCallingAgent — no results fallback
# ═══════════════════════════════════════════════════════════════════════════

def test_agent_no_results_fallback():
    agent = ToolCallingAgent(max_iterations=5)
    res = agent.run("Có xe điện nào giá dưới 200 triệu không?")
    assert res["status"] == "completed"
    assert "không tìm thấy" in res["answer"].lower() or "rất tiếc" in res["answer"].lower()


def test_ticket_preserves_existing_data(isolated_raw_data):
    path = isolated_raw_data / "support_tickets.json"
    before = json.loads(path.read_text(encoding="utf-8"))
    first = submit_support_ticket("Khách A", "Xe bị lỗi", "high")
    second = submit_support_ticket("Khách B", "Phòng bị ẩm mốc")
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after[:-2] == before
    assert first["ticket_id"] != second["ticket_id"]
    assert after[-1]["customer_name"] == "Khách B"
    assert after[-1]["created_at"].endswith("+07:00")


def test_agent_handles_both_tools(isolated_raw_data):
    queries = json.loads((Path(__file__).resolve().parents[1] / "raw-data" / "customer_queries.json").read_text(encoding="utf-8"))
    case = next(case for case in queries if case["id"] == "TC03")
    result = ToolCallingAgent().run(case["query"])
    assert result["status"] == "completed"
    assert result["iterations"] == 2
    actions = [entry for entry in result["trace"] if entry.get("action") in TOOL_MAP]
    assert [entry["action"] for entry in actions] == case["expected_tools"]
    assert actions[0]["arguments"] == case["expected_args_catalog"]
    assert actions[1]["arguments"]["customer_name"] == "Phạm Thị Dung"
    assert "Vinpearl" in result["answer"] and "TK-" in result["answer"]
    tickets = json.loads((isolated_raw_data / "support_tickets.json").read_text(encoding="utf-8"))
    assert tickets[-1]["priority"] == "medium"
    assert "ẩm mốc" in tickets[-1]["issue_description"]


def test_iteration_limit_stops_before_ticket(isolated_raw_data):
    path = isolated_raw_data / "support_tickets.json"
    before = path.read_bytes()
    result = ToolCallingAgent(max_iterations=1).run(
        "Cho tôi xem xe điện dưới 600 triệu. Tôi tên Khách A, xe bị lỗi, cần xử lý gấp."
    )
    assert result["status"] == "max_iterations_reached"
    assert result["iterations"] == 1
    assert "VinFast" in result["answer"]
    assert path.read_bytes() == before
    assert all(entry.get("action") != "submit_support_ticket" for entry in result["trace"])


def test_zero_iteration_limit_calls_no_tools():
    result = ToolCallingAgent(max_iterations=0).run("Xem xe điện dưới 600 triệu")
    assert result["status"] == "max_iterations_reached"
    assert result["iterations"] == 0
    assert len(result["trace"]) == 1


@pytest.mark.parametrize("budget, expected", [
    ("600 triệu", 600000000), ("1,2 tỷ", 1200000000),
    ("1 tỷ 200 triệu", 1200000000), ("600.000.000 đồng", 600000000),
])
def test_vietnamese_budget_is_used_by_catalog(budget, expected):
    result = ToolCallingAgent().run(f"Tìm xe điện giá dưới {budget}")
    assert result["status"] == "completed"
    assert result["trace"][1]["arguments"]["max_price"] == expected
    assert all(p["price_vnd"] <= expected for p in result["trace"][1]["observation"])


def test_missing_customer_name_does_not_create_ticket(isolated_raw_data):
    path = isolated_raw_data / "support_tickets.json"
    before = path.read_bytes()
    result = ToolCallingAgent().run("Xe VinFast bị lỗi, cần xử lý gấp")
    assert result["status"] == "needs_input"
    assert path.read_bytes() == before


def test_missing_catalog_reports_tool_error(isolated_raw_data):
    (isolated_raw_data / "product_catalog.json").unlink()
    result = ToolCallingAgent().run("Tìm xe điện dưới 600 triệu")
    assert result["status"] == "tool_error"
    assert "không tìm thấy sản phẩm" not in result["answer"].lower()


def test_invalid_catalog_reports_tool_error(isolated_raw_data):
    (isolated_raw_data / "product_catalog.json").write_text("invalid JSON", encoding="utf-8")
    result = ToolCallingAgent().run("Tìm xe điện dưới 600 triệu")
    assert result["status"] == "tool_error"


def test_faq_does_not_call_tools():
    result = ToolCallingAgent().run("Chính sách bảo hành pin xe điện VinFast kéo dài bao lâu?")
    assert result["status"] == "completed"
    assert all(entry.get("action") not in TOOL_MAP for entry in result["trace"])


def test_trace_resets_between_requests():
    agent = ToolCallingAgent()
    first = agent.run("Tìm xe điện dưới 600 triệu")
    second = agent.run("Chính sách bảo hành VinFast?")
    assert first["trace"] is not second["trace"]
    assert all(entry.get("action") != "search_product_catalog" for entry in second["trace"])


def test_out_of_scope_request():
    result = ToolCallingAgent().run("Cho tôi xem thời tiết Hà Nội")
    assert result["status"] == "completed"
    assert "Vingroup" in result["answer"]
    assert all(entry.get("action") not in TOOL_MAP for entry in result["trace"])


def test_out_of_scope_issue_does_not_create_ticket(isolated_raw_data):
    path = isolated_raw_data / "support_tickets.json"
    before = path.read_bytes()
    result = ToolCallingAgent().run("Tôi tên Khách A, máy tính của tôi bị lỗi, cần xử lý gấp.")
    assert "Vingroup" in result["answer"]
    assert path.read_bytes() == before


def test_non_urgent_ticket_has_low_priority(isolated_raw_data):
    result = ToolCallingAgent().run("Tôi tên Khách A, xe VinFast bị lỗi, nhưng không gấp.")
    assert result["status"] == "completed"
    tickets = json.loads((isolated_raw_data / "support_tickets.json").read_text(encoding="utf-8"))
    assert tickets[-1]["priority"] == "low"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
