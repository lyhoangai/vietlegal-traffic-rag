from unittest.mock import AsyncMock, patch

import pytest

from tests.agent.support import _make_state


@pytest.mark.asyncio
async def test_generator_speed_returns_missing_evidence_when_docs_missing():
    from src.agent.answers import generator

    state = _make_state("toc do toi da")
    state["intent"] = "speed"
    state["collection_used"] = "traffic_speed"
    result = await generator(state)
    assert result["confidence"] == 0.3
    assert "toc do" in result["answer"].lower() or "tốc độ" in result["answer"].lower()


@pytest.mark.asyncio
async def test_generator_speed_can_return_rule_based_expressway_answer():
    from src.agent.answers import generator

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
    from src.agent.answers import generator

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
    from src.agent.answers import generator

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
    from src.agent.answers import generator

    state = _make_state("toc do toi da")
    state["intent"] = "speed"
    state["collection_used"] = "traffic_speed"
    state["sources"] = ["thong_tu_38_2024_bgtvt trang 12"]
    state["reranked_docs"] = ["Dieu 12. Toc do toi da ..."]

    with patch(
        "src.agent.answers.invoke_with_fallback",
        new=AsyncMock(return_value="Theo Khoản 2 Điều 12, tốc độ tối đa là 80 km/h."),
    ):
        result = await generator(state)

    assert result["confidence"] >= 0.5
    lowered = result["answer"].lower()
    assert "điều 12" in lowered or "dieu 12" in lowered


@pytest.mark.asyncio
async def test_generator_penalty_vague_answer_falls_back_to_safe_response():
    from src.agent.answers import generator

    state = _make_state("o to vuot den do phat bao nhieu")
    state["intent"] = "penalty"
    state["collection_used"] = "traffic_penalties"
    state["sources"] = ["nghi_dinh_168_2024 trang 9"]
    state["reranked_docs"] = ["Khoan 5 Dieu ..."]
    with patch(
        "src.agent.answers.invoke_with_fallback",
        new=AsyncMock(return_value="Khong co thong tin cu the ve muc phat."),
    ):
        result = await generator(state)

    assert result["confidence"] <= 0.4
    assert "hien chua du can cu" in result["answer"].lower() or "hiện chưa đủ căn cứ" in result["answer"].lower()


@pytest.mark.asyncio
async def test_generator_out_of_scope_returns_scope_limitation():
    from src.agent.answers import generator

    result = await generator(_make_state("doi bang lai xe can giay to gi"))
    assert result["confidence"] == 0.2
    assert "pham vi demo" in result["answer"].lower() or "phạm vi demo" in result["answer"].lower()


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
    from src.agent.answers import generator

    with patch(
        "src.agent.answers.invoke_with_fallback",
        new=AsyncMock(side_effect=AssertionError("should not call llm")),
    ):
        result = await generator(_make_state(query))

    assert result["confidence"] == 0.2
    assert "pham vi demo" in result["answer"].lower() or "phạm vi demo" in result["answer"].lower()


@pytest.mark.asyncio
async def test_generator_penalty_red_light_car_returns_rule_based_answer():
    from src.agent.answers import generator

    state = _make_state("o to vuot den do phat bao nhieu")
    state["intent"] = "penalty"
    state["entities"] = {"vehicle_type": "ô tô", "violation_type": "vượt đèn đỏ"}
    state["sources"] = ["nghi_dinh_168_2024 trang 16", "nghi_dinh_168_2024 trang 18"]
    state["reranked_docs"] = [
        (
            "9. Phat tien tu 18.000.000 dong den 20.000.000 dong doi voi nguoi dieu khien xe "
            "thuc hien mot trong cac hanh vi vi pham sau day: b) Khong chap hanh hieu lenh cua den tin hieu giao thong."
        ),
        "16. ... diem b khoan 9 Dieu nay bi tru diem giay phep lai xe 04 diem.",
    ]

    result = await generator(state)

    assert result["confidence"] >= 0.9
    assert "18.000.000" in result["answer"]
    assert "Căn cứ" in result["answer"]
    assert "trừ 04 điểm" in result["answer"] or "tru 04 diem" in result["answer"].lower()


@pytest.mark.asyncio
async def test_generator_penalty_red_light_motorbike_returns_rule_based_answer():
    from src.agent.answers import generator

    state = _make_state("xe may vuot den do phat bao nhieu")
    state["intent"] = "penalty"
    state["entities"] = {"vehicle_type": "xe máy", "violation_type": "vượt đèn đỏ"}
    state["sources"] = ["nghi_dinh_168_2024 trang 22", "nghi_dinh_168_2024 trang 25"]
    state["reranked_docs"] = [
        (
            "7. Phat tien tu 4.000.000 dong den 6.000.000 dong doi voi nguoi dieu khien xe "
            "thuc hien mot trong cac hanh vi vi pham sau day: c) Khong chap hanh hieu lenh cua den tin hieu giao thong."
        ),
        "13. ... diem c, diem d, diem d khoan 7 Dieu nay bi tru diem giay phep lai xe 04 diem.",
    ]

    result = await generator(state)

    assert result["confidence"] >= 0.9
    assert "4.000.000" in result["answer"]
    assert "Căn cứ" in result["answer"]
    assert "04" in result["answer"]


@pytest.mark.asyncio
async def test_generator_penalty_red_light_follow_up_uses_history_context():
    from src.agent.answers import generator

    state = _make_state("the xe may thi sao?")
    state["intent"] = "penalty"
    state["messages"] = [
        {"role": "user", "content": "o to vuot den do phat bao nhieu"},
        {"role": "assistant", "content": "Theo diem b khoan 9 Dieu 6 ..."},
    ]
    state["entities"] = {"vehicle_type": "xe máy", "violation_type": None}
    state["sources"] = ["nghi_dinh_168_2024 trang 22", "nghi_dinh_168_2024 trang 25"]
    state["reranked_docs"] = [
        (
            "7. Phat tien tu 4.000.000 dong den 6.000.000 dong doi voi nguoi dieu khien xe "
            "thuc hien mot trong cac hanh vi vi pham sau day: c) Khong chap hanh hieu lenh cua den tin hieu giao thong."
        ),
        "13. ... diem c, diem d, diem d khoan 7 Dieu nay bi tru diem giay phep lai xe 04 diem.",
    ]

    result = await generator(state)

    assert "4.000.000" in result["answer"]
    assert "Căn cứ" in result["answer"]


@pytest.mark.asyncio
async def test_generator_penalty_red_light_mentions_web_confirmation_when_requested():
    from src.agent.answers import generator

    state = _make_state("doi chieu nguon chinh thong muc phat o to vuot den do")
    state["intent"] = "penalty"
    state["entities"] = {"vehicle_type": "ô tô", "violation_type": "vượt đèn đỏ"}
    state["sources"] = [
        "Web · vbpl.vn · Nghi dinh 168/2024/ND-CP - Bo Cong An",
        "nghi_dinh_168_2024 trang 16",
    ]
    state["reranked_docs"] = [
        (
            "9. Phat tien tu 18.000.000 dong den 20.000.000 dong doi voi nguoi dieu khien xe "
            "thuc hien mot trong cac hanh vi vi pham sau day: b) Khong chap hanh hieu lenh cua den tin hieu giao thong."
        ),
        "16. ... diem b khoan 9 Dieu nay bi tru diem giay phep lai xe 04 diem.",
    ]
    state["web_docs"] = [
        "[Nguon web chinh thong]\nTieu de: Nghi dinh 168/2024/ND-CP\nURL: https://vbpl.vn\nTom tat: ... vuot den do ..."
    ]

    result = await generator(state)

    assert "18.000.000" in result["answer"]
    assert "Đã đối chiếu nguồn web chính thống" in result["answer"] or "doi chieu nguon web chinh thong" in result["answer"].lower()


@pytest.mark.asyncio
async def test_generator_penalty_new_alcohol_query_does_not_return_red_light_answer_from_history():
    from src.agent.answers import generator

    state = _make_state("xe tai uong ruou co bi phat khong")
    state["intent"] = "penalty"
    state["messages"] = [
        {"role": "user", "content": "o to vuot den do phat bao nhieu"},
        {"role": "assistant", "content": "Muc phat tu 18.000.000 dong den 20.000.000 dong."},
    ]
    state["entities"] = {"vehicle_type": "xe tải", "violation_type": None}
    state["sources"] = ["Web · vbpl.vn · Nghi dinh 168/2024/ND-CP", "nghi_dinh_168_2024 trang 16"]
    state["reranked_docs"] = [
        "Dieu 6 ... Khong chap hanh hieu lenh den tin hieu giao thong.",
        "Dieu 8. Xe tai chi can co nong do con khi dieu khien xe la da bi xu phat. Muc 1: phat tu 6.000.000 dong den 8.000.000 dong.",
    ]
    state["web_docs"] = ["[Nguon web chinh thong]\nTieu de: Nghi dinh 168/2024/ND-CP\nURL: https://vbpl.vn\nTom tat: ... nong do con ..."]

    with patch(
        "src.agent.answers.invoke_with_fallback",
        new=AsyncMock(side_effect=AssertionError("should not call llm")),
    ):
        result = await generator(state)

    assert result["confidence"] >= 0.9
    assert "nồng độ cồn" in result["answer"].lower() or "nong do con" in result["answer"].lower()
    assert "vượt đèn đỏ" not in result["answer"].lower()


@pytest.mark.asyncio
async def test_generator_penalty_helmet_returns_compact_answer_with_web_confirmation():
    from src.agent.answers import generator

    state = _make_state("xe may khong doi mu bao hiem bi xu phat the nao")
    state["intent"] = "penalty"
    state["entities"] = {"vehicle_type": "xe máy", "violation_type": "không đội mũ bảo hiểm"}
    state["sources"] = ["Web · vbpl.vn · Nghi dinh 168/2024/ND-CP", "nghi_dinh_168_2024 trang 20"]
    state["reranked_docs"] = [
        "Dieu 7 ... 2. Phat tien tu 400.000 dong den 600.000 dong doi voi nguoi dieu khien xe ...",
        "h) Khong doi mu bao hiem ... 2. Phat tien tu 400.000 dong den 600.000 dong ...",
    ]
    state["web_docs"] = ["[Nguon web chinh thong]\nTieu de: Nghi dinh 168/2024/ND-CP\nURL: https://vbpl.vn\nTom tat: ... mu bao hiem ..."]

    result = await generator(state)

    assert result["confidence"] >= 0.9
    assert "400.000" in result["answer"]
    assert "Điều 7" in result["answer"] or "dieu 7" in result["answer"].lower()
    assert "Đã đối chiếu nguồn web chính thống" in result["answer"] or "doi chieu nguon web chinh thong" in result["answer"].lower()


@pytest.mark.asyncio
async def test_generator_penalty_helmet_follow_up_for_electric_motorbike_does_not_require_web_docs():
    from src.agent.answers import generator

    state = _make_state("neu la xe may dien thi sao")
    state["intent"] = "penalty"
    state["messages"] = [
        {"role": "user", "content": "Xe may khong doi mu bao hiem thi phat bao nhieu?"},
        {"role": "assistant", "content": "Xe may khong doi mu bao hiem bi phat tu 400.000 dong den 600.000 dong."},
    ]
    state["entities"] = {"vehicle_type": "xe máy", "violation_type": None}
    state["sources"] = ["nghi_dinh_168_2024 trang 20"]
    state["reranked_docs"] = ["Dieu 7 quy dinh ve mu bao hiem cho nguoi di xe may. 400.000 dong den 600.000 dong."]

    result = await generator(state)

    assert result["confidence"] >= 0.9
    assert "400.000" in result["answer"]
    assert "168/2024/NĐ-CP" in result["answer"] or "168/2024/nd-cp" in result["answer"].lower()


@pytest.mark.asyncio
async def test_generator_penalty_alcohol_car_follow_up_returns_compact_answer_with_web_confirmation():
    from src.agent.answers import generator

    state = _make_state("o to")
    state["intent"] = "penalty"
    state["messages"] = [
        {"role": "user", "content": "nong do con bao nhieu thi bi phat"},
        {"role": "assistant", "content": "De tra cuu chinh xac, anh/chi dang di xe gi?"},
    ]
    state["entities"] = {"vehicle_type": "ô tô", "violation_type": "nồng độ cồn"}
    state["sources"] = ["Web · vbpl.vn · Nghi dinh 168/2024/ND-CP", "nghi_dinh_168_2024 trang 16"]
    state["reranked_docs"] = [
        "Dieu 6 ... 6. Phat tien tu 6.000.000 dong den 8.000.000 dong ... nong do con ...",
        "9. Phat tien tu 18.000.000 dong den 20.000.000 dong ... nong do con ...",
        "11. Phat tien tu 30.000.000 dong den 40.000.000 dong ... nong do con ...",
    ]
    state["web_docs"] = ["[Nguon web chinh thong]\nTieu de: Nghi dinh 168/2024/ND-CP\nURL: https://vbpl.vn\nTom tat: ... nong do con ..."]

    result = await generator(state)

    assert result["confidence"] >= 0.9
    assert "6.000.000" in result["answer"]
    assert "18.000.000" in result["answer"]
    assert "30.000.000" in result["answer"]
    assert "Đã đối chiếu nguồn web chính thống" in result["answer"] or "doi chieu nguon web chinh thong" in result["answer"].lower()


def test_build_generator_prompt_includes_local_and_web_contexts():
    from src.agent.answers import build_generator_prompt

    state = _make_state("van ban moi nhat")
    state["retrieved_docs"] = ["Tai lieu cuc bo"]
    state["web_docs"] = ["[Official web source]\nTitle: Van ban moi\nSummary: Noi dung moi"]

    prompt = build_generator_prompt(state)

    assert "Tai lieu cuc bo" in prompt
    assert "Nguồn web chính thống" in prompt or "Nguon web chinh thong" in prompt
    assert "Van ban moi" in prompt


def test_build_early_answer_speed_requires_evidence_docs():
    from src.agent.answers import build_early_answer

    state = _make_state("toc do toi da tren duong cao toc la bao nhieu")
    state["intent"] = "speed"
    state["collection_used"] = "traffic_speed"

    answer, confidence = build_early_answer(state, [])

    assert answer is not None
    assert "tốc độ hoặc khoảng cách an toàn" in answer.lower() or "toc do hoac khoang cach an toan" in answer.lower()
    assert confidence == 0.3


def test_build_early_answer_helmet_requires_matching_evidence_docs():
    from src.agent.answers import build_early_answer

    state = _make_state("xe may khong doi mu bao hiem bi xu phat the nao")
    state["intent"] = "penalty"
    state["entities"] = {"vehicle_type": "xe may", "violation_type": "khong doi mu bao hiem"}
    state["retrieved_docs"] = ["Dieu 12 quy dinh ve toc do toi da tren cao toc."]
    state["reranked_docs"] = ["Dieu 12 quy dinh ve toc do toi da tren cao toc."]

    answer, confidence = build_early_answer(state, state["reranked_docs"])

    assert answer is None
    assert confidence is None
