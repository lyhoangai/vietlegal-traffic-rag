from unittest.mock import AsyncMock, patch

import pytest

from tests.agent.support import _make_state

async def test_intent_analyzer_returns_dict():
    from src.agent.intent import intent_analyzer

    with patch(
        "src.agent.intent.invoke_with_fallback",
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
    from src.agent.intent import intent_analyzer

    with patch(
        "src.agent.intent.invoke_with_fallback",
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
    from src.agent.intent import intent_analyzer

    with patch(
        "src.agent.intent.invoke_with_fallback",
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
    from src.agent.intent import intent_analyzer

    with patch(
        "src.agent.intent.invoke_with_fallback",
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
    from src.agent.intent import intent_analyzer

    with patch(
        "src.agent.intent.invoke_with_fallback",
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
    from src.agent.intent import intent_analyzer

    with patch(
        "src.agent.intent.invoke_with_fallback",
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
    from src.agent.intent import intent_analyzer

    with patch(
        "src.agent.intent.invoke_with_fallback",
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
    from src.agent.intent import intent_analyzer

    with patch(
        "src.agent.intent.invoke_with_fallback",
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

def test_infer_entities_handles_vietnamese_diacritics_for_den_do():
    from src.agent.intent import _infer_entities_from_query

    query = "ô tô vượt đèn đỏ phạt bao nhiêu"
    entities = _infer_entities_from_query(
        query,
        {"vehicle_type": None, "violation_type": None},
    )

    assert entities["vehicle_type"] == "ô tô"
    assert entities["violation_type"] == "vượt đèn đỏ"

