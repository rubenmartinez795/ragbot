import pandas as pd
from langsmith import Client

from ragbot.utils import utils


def main():
    # Use LangChain project
    utils.use_langsmith("example")

    # QA
    inputs = [
        "q1",
        "q2",
        "q3",
    ]

    outputs = [
        "a1",
        "a2",
        "a3",
    ]

    # Dataset
    qa_pairs = [{"input": q, "output": a} for q, a in zip(inputs, outputs)]
    df = pd.DataFrame(qa_pairs)

    client = Client()
    dataset_name = "ds"

    # Store
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Description",
    )
    client.create_examples(
        inputs=[{"input": q} for q in inputs],
        outputs=[{"output": a} for a in outputs],
        dataset_id=dataset.id,
    )


if __name__ == "__main__":
    main()
