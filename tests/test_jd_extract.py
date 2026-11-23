"""The extraction call: its prompt, and what is made of the model's answer."""

from llmfakes import ScriptedLLM

from tailor.jd import extract_job

ANSWER = {
    "title": " Backend Engineer ",
    "company": "",
    "responsibilities": ["Run Go services", "Run Go services", " ", "Own on-call"],
    "requirements": ["PostgreSQL"],
    "nice_to_have": [],
    "keywords": ["Go", "Kafka"],
}


def test_the_answer_is_cleaned_and_the_source_fills_what_the_model_left_empty() -> None:
    llm = ScriptedLLM([ANSWER])
    jd = extract_job(llm, "posting text", model="m", company_hint="Northwind")
    assert jd.title == "Backend Engineer"
    assert jd.company == "Northwind"
    assert jd.responsibilities == ("Run Go services", "Own on-call")
    assert jd.keywords == ("Go", "Kafka")


def test_one_call_whose_schema_names_all_six_fields_and_whose_prompt_holds_the_posting() -> None:
    llm = ScriptedLLM([ANSWER])
    extract_job(llm, "the posting body", model="m")
    assert len(llm.calls) == 1
    messages, schema = llm.calls[0]
    assert "the posting body" in messages[-1]["content"]
    assert set(schema["properties"]) == {
        "title",
        "company",
        "responsibilities",
        "requirements",
        "nice_to_have",
        "keywords",
    }
    assert schema["additionalProperties"] is False
