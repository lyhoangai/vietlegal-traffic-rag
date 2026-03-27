"""Unit tests for agent nodes."""

from unittest.mock import AsyncMock, patch

import pytest


def _make_state(query: str) -> dict:
    return {
        "messages": [],
        "user_query": query,
        "intent": "general",
        "entities": {},
        "retrieved_docs": [],
        "reranked_docs": [],
        "web_docs": [],
        "sources": [],
        "needs_clarification": False,
        "clarification_question": "",
        "answer": "",
        "confidence": 0.0,
        "llm_provider": "gemini",
        "collection_used": "traffic_law",
    }


@pytest.mark.asyncio
async def test_query_router_penalty():
    from src.agent.nodes import query_router

    state = _make_state("ô tô vượt đèn đỏ phạt bao nhiêu")
    state["intent"] = "penalty"
    result = await query_router(state)
    assert result["collection_used"] == "traffic_penalties"


@pytest.mark.asyncio
async def test_query_router_speed():
    from src.agent.nodes import query_router

    state = _make_state("tốc độ tối đa trên đường cao tốc")
    state["intent"] = "speed"
    result = await query_router(state)
    assert result["collection_used"] == "traffic_speed"


@pytest.mark.asyncio
async def test_intent_analyzer_returns_dict():
    from src.agent.nodes import intent_analyzer

    with patch(
        "src.agent.nodes.invoke_with_fallback",
        new=AsyncMock(
            return_value=(
                '{"intent": "penalty", "vehicle_type": "ô tô", '
                '"violation_type": "vượt đèn đỏ", "alcohol_level": null, '
                '"speed_value": null, "needs_clarification": false, "missing_field": null}'
            )
        ),
    ):
        state = _make_state("ô tô vượt đèn đỏ phạt bao nhiêu?")
        result = await intent_analyzer(state)
    assert result["intent"] == "penalty"
    assert result["entities"]["vehicle_type"] == "ô tô"
    assert result["needs_clarification"] is False


@pytest.mark.asyncio
async def test_intent_analyzer_needs_clarification():
    from src.agent.nodes import intent_analyzer

    with patch(
        "src.agent.nodes.invoke_with_fallback",
        new=AsyncMock(
            return_value=(
                '{"intent": "penalty", "vehicle_type": null, '
                '"violation_type": "vượt đèn đỏ", "alcohol_level": null, '
                '"speed_value": null, "needs_clarification": true, "missing_field": "vehicle_type"}'
            )
        ),
    ):
        state = _make_state("vượt đèn đỏ phạt bao nhiêu?")
        result = await intent_analyzer(state)
    assert result["needs_clarification"] is True


@pytest.mark.asyncio
async def test_intent_analyzer_out_of_scope_skips_clarification():
    from src.agent.nodes import intent_analyzer

    with patch(
        "src.agent.nodes.invoke_with_fallback",
        new=AsyncMock(
            return_value=(
                '{"intent": "penalty", "vehicle_type": null, '
                '"violation_type": null, "alcohol_level": null, '
                '"speed_value": null, "needs_clarification": true, "missing_field": "vehicle_type"}'
            )
        ),
    ):
        state = _make_state("doi bang lai xe can giay to gi")
        result = await intent_analyzer(state)

    assert result["needs_clarification"] is False
    assert result["clarification_question"] == ""


@pytest.mark.asyncio
async def test_intent_analyzer_history_fills_missing_entities_without_overriding_current_query():
    from src.agent.nodes import intent_analyzer

    with patch(
        "src.agent.nodes.invoke_with_fallback",
        new=AsyncMock(
            return_value=(
                '{"intent": "penalty", "vehicle_type": null, '
                '"violation_type": null, "alcohol_level": null, '
                '"speed_value": null, "needs_clarification": false, "missing_field": null}'
            )
        ),
    ):
        state = _make_state("thế xe máy thì sao?")
        state["messages"] = [
            {"role": "user", "content": "ô tô vượt đèn đỏ phạt bao nhiêu"},
            {"role": "assistant", "content": "Mức phạt ..."},
        ]
        result = await intent_analyzer(state)

    assert result["entities"]["vehicle_type"] == "xe máy"
    assert result["entities"]["violation_type"] == "vượt đèn đỏ"


@pytest.mark.asyncio
async def test_intent_analyzer_current_query_can_override_previous_violation():
    from src.agent.nodes import intent_analyzer

    with patch(
        "src.agent.nodes.invoke_with_fallback",
        new=AsyncMock(
            return_value=(
                '{"intent": "penalty", "vehicle_type": null, '
                '"violation_type": null, "alcohol_level": null, '
                '"speed_value": null, "needs_clarification": false, "missing_field": null}'
            )
        ),
    ):
        state = _make_state("còn đi sai làn thì sao?")
        state["messages"] = [
            {"role": "user", "content": "ô tô vượt đèn đỏ phạt bao nhiêu"},
            {"role": "assistant", "content": "Mức phạt ..."},
        ]
        result = await intent_analyzer(state)

    assert result["entities"]["violation_type"] == "đi sai làn"


@pytest.mark.asyncio
async def test_intent_analyzer_new_alcohol_query_does_not_inherit_red_light_violation_from_history():
    from src.agent.nodes import intent_analyzer

    with patch(
        "src.agent.nodes.invoke_with_fallback",
        new=AsyncMock(return_value="not-json"),
    ):
        state = _make_state("xe tai uong ruou co bi phat khong")
        state["messages"] = [
            {"role": "user", "content": "o to vuot den do phat bao nhieu"},
            {"role": "assistant", "content": "Muc phat tu 18.000.000 dong den 20.000.000 dong."},
        ]
        result = await intent_analyzer(state)

    assert result["intent"] == "penalty"
    assert result["entities"]["vehicle_type"] == "xe tải"
    assert result["entities"]["violation_type"] == "nồng độ cồn"


@pytest.mark.asyncio
async def test_intent_analyzer_falls_back_to_penalty_intent_from_follow_up_history_when_llm_output_is_invalid():
    from src.agent.nodes import intent_analyzer

    with patch(
        "src.agent.nodes.invoke_with_fallback",
        new=AsyncMock(return_value="not-json"),
    ):
        state = _make_state("neu la xe may dien thi sao?")
        state["messages"] = [
            {"role": "user", "content": "xe may khong doi mu bao hiem thi phat bao nhieu"},
            {"role": "assistant", "content": "xe may khong doi mu bao hiem bi phat tu 400.000 dong den 600.000 dong"},
        ]
        result = await intent_analyzer(state)

    assert result["intent"] == "penalty"
    assert "xe may" in result["entities"]["vehicle_type"].lower().replace("á", "a")


@pytest.mark.asyncio
async def test_intent_analyzer_falls_back_to_history_heuristics_when_llm_provider_errors():
    from src.agent.nodes import intent_analyzer

    with patch(
        "src.agent.nodes.invoke_with_fallback",
        new=AsyncMock(side_effect=RuntimeError("rate limit")),
    ):
        state = _make_state("neu la xe may dien thi sao?")
        state["messages"] = [
            {"role": "user", "content": "xe may khong doi mu bao hiem thi phat bao nhieu"},
            {"role": "assistant", "content": "xe may khong doi mu bao hiem bi phat tu 400.000 dong den 600.000 dong"},
        ]
        result = await intent_analyzer(state)

    assert result["intent"] == "penalty"
    assert "xe may" in result["entities"]["vehicle_type"].lower().replace("á", "a")
    assert result["needs_clarification"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_vehicle", "expected_violation"),
    [
        (
            "xe con vuot den do phat the nao",
            "\u00f4 t\u00f4",
            "v\u01b0\u1ee3t \u0111\u00e8n \u0111\u1ecf",
        ),
        (
            "xe gan may vuot den do phat the nao",
            "xe m\u00e1y",
            "v\u01b0\u1ee3t \u0111\u00e8n \u0111\u1ecf",
        ),
        (
            "xe mo to khong doi mu bao hiem phat bao nhieu",
            "xe m\u00e1y",
            "kh\u00f4ng \u0111\u1ed9i m\u0169 b\u1ea3o hi\u1ec3m",
        ),
    ],
)
async def test_intent_analyzer_maps_benchmark_vehicle_synonyms_when_llm_is_unavailable(
    query,
    expected_vehicle,
    expected_violation,
):
    from src.agent.nodes import intent_analyzer

    with patch(
        "src.agent.nodes.invoke_with_fallback",
        new=AsyncMock(side_effect=RuntimeError("401 unauthorized")),
    ):
        result = await intent_analyzer(_make_state(query))

    assert result["intent"] == "penalty"
    assert result["entities"]["vehicle_type"] == expected_vehicle
    assert result["entities"]["violation_type"] == expected_violation


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "tai xe o to co con khi lai xe phat bao nhieu",
        "muc phat o to o nguong con thap nhat la bao nhieu",
    ],
)
async def test_intent_analyzer_detects_alcohol_penalty_synonyms_when_llm_is_unavailable(query):
    from src.agent.nodes import intent_analyzer

    with patch(
        "src.agent.nodes.invoke_with_fallback",
        new=AsyncMock(side_effect=RuntimeError("401 unauthorized")),
    ):
        result = await intent_analyzer(_make_state(query))

    assert result["intent"] == "penalty"
    assert result["entities"]["vehicle_type"] == "\u00f4 t\u00f4"
    assert result["entities"]["violation_type"] == "n\u1ed3ng \u0111\u1ed9 c\u1ed3n"


@pytest.mark.asyncio
async def test_reranker_reduces_to_5():
    from src.agent.nodes import reranker

    state = _make_state("test")
    state["retrieved_docs"] = [f"doc {i}" for i in range(20)]
    with patch("src.agent.nodes._reranker_model", None):
        result = await reranker(state)
    assert len(result["reranked_docs"]) == 5


@pytest.mark.asyncio
async def test_reranker_can_be_disabled_via_env(monkeypatch):
    from src.agent.nodes import reranker

    state = _make_state("test")
    state["retrieved_docs"] = [f"doc {i}" for i in range(8)]
    monkeypatch.setenv("ENABLE_RERANKER", "false")

    with patch("src.agent.nodes._get_reranker", side_effect=AssertionError("should not load")):
        result = await reranker(state)

    assert result["reranked_docs"] == state["retrieved_docs"][:5]


@pytest.mark.asyncio
async def test_generator_speed_returns_missing_evidence_when_docs_missing():
    from src.agent.nodes import generator

    state = _make_state("tốc độ tối đa")
    state["intent"] = "speed"
    state["collection_used"] = "traffic_speed"
    result = await generator(state)
    assert result["confidence"] == 0.3
    assert "tốc độ hoặc khoảng cách an toàn" in result["answer"].lower()


def test_build_early_answer_speed_requires_evidence_docs():
    from src.agent.nodes import build_early_answer

    state = _make_state("toc do toi da tren duong cao toc la bao nhieu")
    state["intent"] = "speed"
    state["collection_used"] = "traffic_speed"

    answer, confidence = build_early_answer(state, [])

    assert answer == "Không đủ căn cứ trong tài liệu hiện có để kết luận về tốc độ hoặc khoảng cách an toàn."
    assert confidence == 0.3


def test_build_early_answer_helmet_requires_matching_evidence_docs():
    from src.agent.nodes import build_early_answer

    state = _make_state("xe may khong doi mu bao hiem bi xu phat the nao")
    state["intent"] = "penalty"
    state["entities"] = {"vehicle_type": "xe may", "violation_type": "khong doi mu bao hiem"}
    state["retrieved_docs"] = ["Dieu 12 quy dinh ve toc do toi da tren cao toc."]
    state["reranked_docs"] = ["Dieu 12 quy dinh ve toc do toi da tren cao toc."]

    answer, confidence = build_early_answer(state, state["reranked_docs"])

    assert answer is None
    assert confidence is None


@pytest.mark.asyncio
async def test_generator_speed_can_return_rule_based_expressway_answer():
    from src.agent.nodes import generator

    state = _make_state("toc do toi da tren duong cao toc la bao nhieu")
    state["intent"] = "speed"
    state["collection_used"] = "traffic_speed"
    state["sources"] = ["thong_tu_38_2024_bgtvt trang 2"]
    state["reranked_docs"] = [
        "Khoan 2 Dieu 9: toc do khai thac toi da cho phep tren duong cao toc la 120 km/h.",
        "Khoan 3 Dieu 9: toc do khai thac toi thieu tren duong cao toc la 60 km/h.",
    ]

    result = await generator(state)

    assert result["confidence"] >= 0.9
    assert "120 km/h" in result["answer"]
    assert "38/2024/TT-BGTVT" in result["answer"]


@pytest.mark.asyncio
async def test_generator_speed_can_handle_expressway_question_without_explicit_toc_do_phrase():
    from src.agent.nodes import generator

    state = _make_state("cao toc co cho chay 120 km/h khong")
    state["intent"] = "speed"
    state["collection_used"] = "traffic_speed"
    state["sources"] = ["thong_tu_38_2024_bgtvt trang 2"]
    state["reranked_docs"] = [
        "Khoan 2 Dieu 9: toc do khai thac toi da cho phep tren duong cao toc la 120 km/h.",
        "Khoan 3 Dieu 9: toc do khai thac toi thieu tren duong cao toc la 60 km/h.",
    ]

    result = await generator(state)

    assert result["confidence"] >= 0.9
    assert "120 km/h" in result["answer"]
    assert "38/2024/TT-BGTVT" in result["answer"]


@pytest.mark.asyncio
async def test_generator_speed_can_use_history_and_retrieved_docs_when_reranked_docs_are_weak():
    from src.agent.nodes import generator

    state = _make_state("vay nguong chay nhanh nhat la bao nhieu")
    state["intent"] = "speed"
    state["collection_used"] = "traffic_speed"
    state["messages"] = [
        {"role": "user", "content": "Toc do toi thieu tren duong cao toc la bao nhieu?"},
        {"role": "assistant", "content": "Toc do khai thac toi thieu tren duong cao toc la 60 km/h."},
    ]
    state["sources"] = ["thong_tu_38_2024_bgtvt trang 2"]
    state["reranked_docs"] = ["Bang 2 Dieu 8: xe con co the di 80 km/h tren mot so loai duong."]
    state["retrieved_docs"] = [
        "Khoan 2 Dieu 9: toc do khai thac toi da cho phep tren duong cao toc la 120 km/h.",
        "Khoan 3 Dieu 9: toc do khai thac toi thieu tren duong cao toc la 60 km/h.",
    ]

    result = await generator(state)

    assert result["confidence"] >= 0.9
    assert "120 km/h" in result["answer"]
    assert "38/2024/TT-BGTVT" in result["answer"]


@pytest.mark.asyncio
async def test_generator_speed_can_answer_from_available_docs():
    from src.agent.nodes import generator

    state = _make_state("tốc độ tối đa")
    state["intent"] = "speed"
    state["collection_used"] = "traffic_speed"
    state["sources"] = ["thong_tu_38_2024_bgtvt trang 12"]
    state["reranked_docs"] = ["Điều 12. Tốc độ tối đa ..."]

    with patch(
        "src.agent.nodes.invoke_with_fallback",
        new=AsyncMock(return_value="Theo Điều 12, tốc độ tối đa là 80 km/h."),
    ):
        result = await generator(state)

    assert result["confidence"] >= 0.5
    assert "điều 12" in result["answer"].lower()


@pytest.mark.asyncio
async def test_generator_penalty_vague_answer_falls_back_to_safe_response():
    from src.agent.nodes import generator

    state = _make_state("ô tô vượt đèn đỏ phạt bao nhiêu")
    state["intent"] = "penalty"
    state["collection_used"] = "traffic_penalties"
    state["sources"] = ["nghi_dinh_168_2024 trang 9"]
    state["reranked_docs"] = ["Khoản 5 Điều ..."]
    with patch(
        "src.agent.nodes.invoke_with_fallback",
        new=AsyncMock(return_value="Không có thông tin cụ thể về mức phạt."),
    ):
        result = await generator(state)
    assert result["confidence"] <= 0.4
    assert "hiện chưa đủ căn cứ" in result["answer"].lower()


@pytest.mark.asyncio
async def test_generator_out_of_scope_returns_scope_limitation():
    from src.agent.nodes import generator

    state = _make_state("đổi bằng lái xe cần giấy tờ gì")
    result = await generator(state)
    assert result["confidence"] == 0.2
    assert "ngoài phạm vi demo" in result["answer"].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "chuyen quyen so huu o to can lam thu tuc gi",
        "muon mo doanh nghiep van tai can thu tuc gi",
        "gay tai nan phai den bao nhieu tien",
    ],
)
async def test_generator_out_of_scope_handles_benchmark_scope_queries(query):
    from src.agent.nodes import generator

    state = _make_state(query)

    with patch(
        "src.agent.nodes.invoke_with_fallback",
        new=AsyncMock(side_effect=AssertionError("should not call llm")),
    ):
        result = await generator(state)

    assert result["confidence"] == 0.2
    assert "ph\u1ea1m vi demo" in result["answer"].lower()


def test_infer_entities_handles_vietnamese_diacritics_for_den_do():
    from src.agent.nodes import _infer_entities_from_query

    query = "ô tô vượt đèn đỏ phạt bao nhiêu"
    entities = _infer_entities_from_query(
        query,
        {"vehicle_type": None, "violation_type": None},
    )

    assert entities["vehicle_type"] == "ô tô"
    assert entities["violation_type"] == "vượt đèn đỏ"

@pytest.mark.asyncio
async def test_generator_penalty_red_light_car_returns_rule_based_answer():
    from src.agent.nodes import generator

    state = _make_state("o to vuot den do phat bao nhieu")
    state["intent"] = "penalty"
    state["entities"] = {"vehicle_type": "ô tô", "violation_type": "vượt đèn đỏ"}
    state["sources"] = ["nghi_dinh_168_2024 trang 16", "nghi_dinh_168_2024 trang 18"]
    state["reranked_docs"] = [
        (
            "9. Phạt tiền từ 18.000.000 đồng đến 20.000.000 đồng đối với người điều khiển xe "
            "thực hiện một trong các hành vi vi phạm sau đây: b) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông."
        ),
        "16. ... điểm b khoản 9 Điều này bị trừ điểm giấy phép lái xe 04 điểm.",
    ]

    result = await generator(state)

    assert result["confidence"] >= 0.9
    assert "18.000.000 đồng đến 20.000.000 đồng" in result["answer"]
    assert "Căn cứ:" in result["answer"]
    assert "điểm b khoản 9" in result["answer"]
    assert "điểm b khoản 16 Điều 6" in result["answer"]
    assert "trừ 04 điểm" in result["answer"]


@pytest.mark.asyncio
async def test_generator_penalty_red_light_motorbike_returns_rule_based_answer():
    from src.agent.nodes import generator

    state = _make_state("xe may vuot den do phat bao nhieu")
    state["intent"] = "penalty"
    state["entities"] = {"vehicle_type": "xe máy", "violation_type": "vượt đèn đỏ"}
    state["sources"] = ["nghi_dinh_168_2024 trang 22", "nghi_dinh_168_2024 trang 25"]
    state["reranked_docs"] = [
        (
            "7. Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển xe "
            "thực hiện một trong các hành vi vi phạm sau đây: c) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông."
        ),
        "13. ... điểm c, điểm d, điểm đ khoản 7 Điều này bị trừ điểm giấy phép lái xe 04 điểm.",
    ]

    result = await generator(state)

    assert result["confidence"] >= 0.9
    assert "4.000.000 đồng đến 6.000.000 đồng" in result["answer"]
    assert "Căn cứ:" in result["answer"]
    assert "điểm c khoản 7" in result["answer"]
    assert "điểm b khoản 13 Điều 7" in result["answer"]
    assert "trừ 04 điểm" in result["answer"]

@pytest.mark.asyncio
async def test_generator_penalty_red_light_follow_up_uses_history_context():
    from src.agent.nodes import generator

    state = _make_state("the xe may thi sao?")
    state["intent"] = "penalty"
    state["messages"] = [
        {"role": "user", "content": "o to vuot den do phat bao nhieu"},
        {"role": "assistant", "content": "Theo điểm b khoản 9 Điều 6 ..."},
    ]
    state["entities"] = {"vehicle_type": "xe máy", "violation_type": None}
    state["sources"] = ["nghi_dinh_168_2024 trang 22", "nghi_dinh_168_2024 trang 25"]
    state["reranked_docs"] = [
        (
            "7. Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển xe "
            "thực hiện một trong các hành vi vi phạm sau đây: c) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông."
        ),
        "13. ... điểm c, điểm d, điểm đ khoản 7 Điều này bị trừ điểm giấy phép lái xe 04 điểm.",
    ]

    result = await generator(state)

    assert "4.000.000 đồng đến 6.000.000 đồng" in result["answer"]
    assert "Căn cứ:" in result["answer"]
    assert "điểm c khoản 7" in result["answer"]


@pytest.mark.asyncio
async def test_generator_penalty_red_light_mentions_web_confirmation_when_requested():
    from src.agent.nodes import generator

    state = _make_state("doi chieu nguon chinh thong muc phat o to vuot den do")
    state["intent"] = "penalty"
    state["entities"] = {"vehicle_type": "ô tô", "violation_type": "vượt đèn đỏ"}
    state["sources"] = [
        "Web · vbpl.vn · Nghị định 168/2024/NĐ-CP - Bộ Công An",
        "nghi_dinh_168_2024 trang 16",
    ]
    state["reranked_docs"] = [
        (
            "9. Phạt tiền từ 18.000.000 đồng đến 20.000.000 đồng đối với người điều khiển xe "
            "thực hiện một trong các hành vi vi phạm sau đây: b) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông."
        ),
        "16. ... điểm b khoản 9 Điều này bị trừ điểm giấy phép lái xe 04 điểm.",
    ]
    state["web_docs"] = [
        "[Nguồn web chính thống]\nTiêu đề: Nghị định 168/2024/NĐ-CP - Bộ Công An\nURL: https://vbpl.vn/bocongan/Pages/vbpq-toanvan.aspx?ItemID=173920\nTóm tắt: ... vượt đèn đỏ ..."
    ]

    result = await generator(state)

    assert "18.000.000 đồng đến 20.000.000 đồng" in result["answer"]
    assert "Đã đối chiếu nguồn web chính thống" in result["answer"]


@pytest.mark.asyncio
async def test_generator_penalty_new_alcohol_query_does_not_return_red_light_answer_from_history():
    from src.agent.nodes import generator

    state = _make_state("xe tai uong ruou co bi phat khong")
    state["intent"] = "penalty"
    state["messages"] = [
        {"role": "user", "content": "o to vuot den do phat bao nhieu"},
        {"role": "assistant", "content": "Muc phat tu 18.000.000 dong den 20.000.000 dong."},
    ]
    state["entities"] = {"vehicle_type": "xe tải", "violation_type": None}
    state["sources"] = [
        "Web · vbpl.vn · Nghị định 168/2024/NĐ-CP - Bộ Công An",
        "nghi_dinh_168_2024 trang 16",
    ]
    state["reranked_docs"] = [
        (
            "Điều 6. Xử phạt người điều khiển xe ô tô ... "
            "9. Phạt tiền từ 18.000.000 đồng đến 20.000.000 đồng đối với hành vi không chấp hành hiệu lệnh đèn tín hiệu giao thông."
        ),
        (
            "Điều 8. Xe tải chỉ cần có nồng độ cồn khi điều khiển xe là đã bị xử phạt. "
            "Mức 1: phạt từ 6.000.000 đồng đến 8.000.000 đồng nếu chưa vượt quá 50 mg/100 ml máu "
            "hoặc chưa vượt quá 0,25 mg/l khí thở."
        ),
    ]
    state["web_docs"] = [
        "[Nguồn web chính thống]\nTiêu đề: Nghị định 168/2024/NĐ-CP - Bộ Công An\nURL: https://vbpl.vn\nTóm tắt: ... nồng độ cồn ..."
    ]

    with patch(
        "src.agent.nodes.invoke_with_fallback",
        new=AsyncMock(side_effect=AssertionError("should not call llm")),
    ):
        result = await generator(state)

    assert result["confidence"] >= 0.9
    assert "nồng độ cồn" in result["answer"].lower()
    assert "vượt đèn đỏ" not in result["answer"].lower()


@pytest.mark.asyncio
async def test_web_searcher_skips_when_local_docs_are_strong():
    from src.agent.nodes import web_searcher

    state = _make_state("quy tac vuot xe tren duong bo")
    state["intent"] = "law"
    state["reranked_docs"] = ["doc 1", "doc 2", "doc 3"]

    with patch("src.agent.nodes.search_official_web", new=AsyncMock(return_value=[])) as mock_search:
        result = await web_searcher(state)

    mock_search.assert_not_awaited()
    assert result["web_docs"] == []


@pytest.mark.asyncio
async def test_web_searcher_uses_web_when_user_explicitly_requests_official_source():
    from src.agent.nodes import web_searcher

    state = _make_state("doi chieu nguon chinh thong quy tac vuot xe")
    state["intent"] = "law"
    state["reranked_docs"] = ["doc 1", "doc 2", "doc 3"]

    with patch(
        "src.agent.nodes.search_official_web",
        new=AsyncMock(
            return_value=[
                {
                    "title": "Luat 36/2024/QH15",
                    "url": "https://vbpl.vn/Pages/vbpq-toanvan.aspx?ItemID=123456",
                    "content": "Quy tắc vượt xe trên đường bộ.",
                    "source": "Web | vbpl.vn | Luat 36/2024/QH15",
                }
            ]
        ),
    ) as mock_search:
        result = await web_searcher(state)

    mock_search.assert_awaited_once()
    assert result["web_docs"]
    assert any("vbpl.vn" in source for source in result["sources"])


@pytest.mark.asyncio
async def test_web_searcher_merges_official_results_when_local_docs_are_missing():
    from src.agent.nodes import web_searcher

    state = _make_state("nghi dinh moi ve toc do toi da")
    state["intent"] = "speed"

    with patch(
        "src.agent.nodes.search_official_web",
        new=AsyncMock(
            return_value=[
                {
                    "title": "Nghi dinh moi",
                    "url": "https://congbao.chinhphu.vn/van-ban/nghi-dinh-moi.htm",
                    "content": "Van ban moi nhat cap nhat toc do toi da.",
                    "source": "Web | congbao.chinhphu.vn | Nghi dinh moi",
                }
            ]
        ),
    ):
        result = await web_searcher(state)

    assert len(result["web_docs"]) == 1
    assert "URL:" in result["web_docs"][0]
    assert "congbao.chinhphu.vn" in result["sources"][-1]


@pytest.mark.asyncio
async def test_web_searcher_uses_web_for_penalty_when_no_rule_based_answer_exists():
    from src.agent.nodes import web_searcher

    state = _make_state("xe may khong doi mu bao hiem bi xu phat the nao")
    state["intent"] = "penalty"
    state["entities"] = {"vehicle_type": "xe máy", "violation_type": "không đội mũ bảo hiểm"}
    state["reranked_docs"] = ["Điểm b khoản 6 ... không đội mũ bảo hiểm ..."]

    with patch(
        "src.agent.nodes.search_official_web",
        new=AsyncMock(
            return_value=[
                {
                    "title": "Nghi dinh 168/2024/ND-CP",
                    "url": "https://vbpl.vn/bocongan/Pages/vbpq-toanvan.aspx?ItemID=173920",
                    "content": "Quy định xử phạt vi phạm hành chính về trật tự, an toàn giao thông.",
                    "source": "Web | vbpl.vn | Nghi dinh 168/2024/ND-CP",
                }
            ]
        ),
    ) as mock_search:
        result = await web_searcher(state)

    mock_search.assert_awaited_once()
    assert result["web_docs"]
    assert any("vbpl.vn" in source for source in result["sources"])


@pytest.mark.asyncio
async def test_web_searcher_can_be_disabled_via_env(monkeypatch):
    from src.agent.nodes import web_searcher

    state = _make_state("nghi dinh moi ve toc do toi da")
    state["intent"] = "speed"
    monkeypatch.setenv("ENABLE_WEB_FALLBACK", "false")

    with patch("src.agent.nodes.search_official_web", new=AsyncMock(return_value=[])) as mock_search:
        result = await web_searcher(state)

    mock_search.assert_not_awaited()
    assert result["web_docs"] == []


@pytest.mark.asyncio
async def test_generator_penalty_helmet_returns_compact_answer_with_web_confirmation():
    from src.agent.nodes import generator

    state = _make_state("xe may khong doi mu bao hiem bi xu phat the nao")
    state["intent"] = "penalty"
    state["entities"] = {"vehicle_type": "xe máy", "violation_type": "không đổi mũ bảo hiểm"}
    state["sources"] = [
        "Web · vbpl.vn · Nghị định 168/2024/NĐ-CP - Bộ Công An",
        "nghi_dinh_168_2024 trang 20",
    ]
    state["reranked_docs"] = [
        (
            "Điều 7. Xử phạt, trừ điểm giấy phép lái của người điều khiển xe mô tô, xe gắn máy ... "
            "2. Phạt tiền từ 400.000 đồng đến 600.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây ..."
        ),
        (
            "h) Không đội “mũ bảo hiểm cho người đi mô tô, xe máy” hoặc đội “mũ bảo hiểm cho người đi mô tô, xe máy” "
            "không cài quai đúng quy cách khi điều khiển xe tham gia giao thông trên đường bộ; "
            "2. Phạt tiền từ 400.000 đồng đến 600.000 đồng đối với người điều khiển xe ..."
        ),
    ]
    state["web_docs"] = [
        "[Nguồn web chính thống]\nTiêu đề: Nghị định 168/2024/NĐ-CP - Bộ Công An\nURL: https://vbpl.vn/bocongan/Pages/vbpq-toanvan.aspx?ItemID=173920\nTóm tắt: ... mũ bảo hiểm cho người đi mô tô, xe máy ..."
    ]

    result = await generator(state)

    assert result["confidence"] >= 0.9
    assert "Xe máy không đội mũ bảo hiểm: phạt từ 400.000 đồng đến 600.000 đồng." in result["answer"]
    assert "Căn cứ: điểm h khoản 2 Điều 7 Nghị định 168/2024/NĐ-CP." in result["answer"]
    assert "Đã đối chiếu nguồn web chính thống" in result["answer"]


@pytest.mark.asyncio
async def test_generator_penalty_helmet_follow_up_for_electric_motorbike_does_not_require_web_docs():
    from src.agent.nodes import generator

    state = _make_state("neu la xe may dien thi sao")
    state["intent"] = "penalty"
    state["messages"] = [
        {"role": "user", "content": "Xe may khong doi mu bao hiem thi phat bao nhieu?"},
        {"role": "assistant", "content": "Xe may khong doi mu bao hiem bi phat tu 400.000 dong den 600.000 dong."},
    ]
    state["entities"] = {"vehicle_type": "xe máy", "violation_type": None}
    state["sources"] = ["nghi_dinh_168_2024 trang 20"]
    state["reranked_docs"] = ["Dieu 7 quy dinh ve mu bao hiem cho nguoi di xe may."]

    result = await generator(state)

    assert result["confidence"] >= 0.9
    assert "400.000" in result["answer"]
    assert "168/2024/NĐ-CP" in result["answer"]


@pytest.mark.asyncio
async def test_generator_penalty_alcohol_car_follow_up_returns_compact_answer_with_web_confirmation():
    from src.agent.nodes import generator

    state = _make_state("o to")
    state["intent"] = "penalty"
    state["messages"] = [
        {"role": "user", "content": "nong do con bao nhieu thi bi phat"},
        {"role": "assistant", "content": "De tra cuu chinh xac, anh/chi dang di xe gi?"},
    ]
    state["entities"] = {"vehicle_type": "ô tô", "violation_type": "nồng độ cồn"}
    state["sources"] = [
        "Web · vbpl.vn · Nghị định 168/2024/NĐ-CP - Bộ Công An",
        "nghi_dinh_168_2024 trang 16",
        "nghi_dinh_168_2024 trang 17",
    ]
    state["reranked_docs"] = [
        (
            "Điều 6. Xử phạt, trừ điểm giấy phép lái xe của người điều khiển xe ô tô ... "
            "6. Phạt tiền từ 6.000.000 đồng đến 8.000.000 đồng đối với người điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn nhưng chưa vượt quá 50 mg/100 ml máu hoặc chưa vượt quá 0,25 mg/l khí thở."
        ),
        (
            "9. Phạt tiền từ 18.000.000 đồng đến 20.000.000 đồng đối với người điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn vượt quá 50 đến 80 mg/100 ml máu hoặc vượt quá 0,25 đến 0,4 mg/l khí thở."
        ),
        (
            "11. Phạt tiền từ 30.000.000 đồng đến 40.000.000 đồng đối với người điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn vượt quá 80 mg/100 ml máu hoặc vượt quá 0,4 mg/l khí thở; "
            "không chấp hành yêu cầu kiểm tra về nồng độ cồn của người thi hành công vụ."
        ),
    ]
    state["web_docs"] = [
        "[Nguồn web chính thống]\nTiêu đề: Nghị định 168/2024/NĐ-CP - Bộ Công An\nURL: https://vbpl.vn/bocongan/Pages/vbpq-toanvan.aspx?ItemID=173920\nTóm tắt: ... nồng độ cồn ... Điều 6 ..."
    ]

    result = await generator(state)

    assert result["confidence"] >= 0.9
    assert "Ô tô chỉ cần có nồng độ cồn khi điều khiển xe là đã bị xử phạt." in result["answer"]
    assert "Mức 1: phạt từ 6.000.000 đồng đến 8.000.000 đồng" in result["answer"]
    assert "Mức 2: phạt từ 18.000.000 đồng đến 20.000.000 đồng" in result["answer"]
    assert "Mức 3: phạt từ 30.000.000 đồng đến 40.000.000 đồng" in result["answer"]
    assert "Căn cứ: điểm c khoản 6, điểm a khoản 9, điểm a và điểm b khoản 11 Điều 6 Nghị định 168/2024/NĐ-CP." in result["answer"]
    assert "Đã đối chiếu nguồn web chính thống" in result["answer"]


def test_build_generator_prompt_includes_local_and_web_contexts():
    from src.agent.nodes import build_generator_prompt

    state = _make_state("van ban moi nhat")
    state["retrieved_docs"] = ["Tai lieu cuc bo"]
    state["web_docs"] = ["[Official web source]\nTitle: Van ban moi\nSummary: Noi dung moi"]

    prompt = build_generator_prompt(state)

    assert "Tai lieu cuc bo" in prompt
    assert "Nguồn web chính thống" in prompt
    assert "Tai lieu cuc bo" in prompt
    assert "Van ban moi" in prompt
