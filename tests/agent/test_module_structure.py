"""Structure regression tests for the split agent modules."""


def test_intent_module_exports_intent_analyzer():
    from src.agent.intent import intent_analyzer

    assert callable(intent_analyzer)


def test_retrieval_module_exports_pipeline_nodes():
    from src.agent.retrieval import query_router, reranker, retriever, web_searcher

    assert callable(query_router)
    assert callable(retriever)
    assert callable(reranker)
    assert callable(web_searcher)


def test_answers_module_exports_generation_helpers():
    from src.agent.answers import (
        build_early_answer,
        build_generator_prompt,
        finalize_generated_answer,
        generator,
    )

    assert callable(build_early_answer)
    assert callable(build_generator_prompt)
    assert callable(finalize_generated_answer)
    assert callable(generator)
