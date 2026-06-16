from openai import OpenAI
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, LLMContextRecall
from rag import answer_with_contexts
from testset import TESTSET

openai_client = OpenAI(timeout=60.0, max_retries=3)

# Run YOUR rag on each question to capture response + retrieved contexts
samples = []
for item in TESTSET:
    print(f"Evaluating: {item['q']}")
    resp, ctxs = answer_with_contexts(item["q"])
    samples.append(SingleTurnSample(
        user_input=item["q"], response=resp,
        retrieved_contexts=ctxs, reference=item["ref"],
    ))

dataset = EvaluationDataset(samples=samples)
judge = llm_factory("gpt-4o-mini", client=openai_client, max_tokens=4096)

result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), LLMContextPrecisionWithoutReference(), LLMContextRecall()],
    llm=judge,
)
print(result)
result.to_pandas().to_csv("eval_hybrid.csv", index=False)