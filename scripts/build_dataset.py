from langsmith import Client

from ragbot.utils import utils


def main():
    # Use LangChain project
    utils.use_langsmith("example")

    # QA
    inputs = [
        "What file formats does Laredo accept for the dataset?",
        "How can I view the metrics of a trained model?",
        "What happens if I select an incompatible algorithm with my problem type?",
        "What do I do if the model takes too long to train?",
    ]

    outputs = [
        "The application accepts files in CSV format.",
        "The metrics are displayed in a table after completing model training, including precision, recall, and F1-score.",
        "If you select an incompatible algorithm, the system will display an error message indicating that you must choose another algorithm that is compatible with the selected problem type.",
        "If the model takes too long to train: check that the dataset size is not excessive, try to reduce the model complexity or use a smaller dataset. Remember that there are models with a large number of parameters that require more time to train.",
    ]

    # Dataset
    client = Client()
    dataset_name = "ds-example"

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
