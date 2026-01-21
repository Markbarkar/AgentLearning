from typing import List
from tqdm import tqdm
from openai import OpenAI
from pymilvus import MilvusClient


class RAG:
    """
    RAG (Retrieval-Augmented Generation) class built upon OpenAI and Milvus.
    """

    def __init__(self, openai_client: OpenAI, milvus_client: MilvusClient):
        self._prepare_openai(openai_client)
        self._prepare_milvus(milvus_client)

    def _emb_text(self, text: str) -> List[float]:
        return (
            self.openai_client.embeddings.create(input=text, model=self.embedding_model)
            .data[0]
            .embedding
        )

    def _prepare_openai(
        self,
        openai_client: OpenAI,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-3.5-turbo",
    ):
        self.openai_client = openai_client
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.SYSTEM_PROMPT = """
Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
"""
        self.USER_PROMPT = """
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""

    def _prepare_milvus(
        self, milvus_client: MilvusClient, collection_name: str = "rag_collection"
    ):
        self.milvus_client = milvus_client
        self.collection_name = collection_name
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)
        embedding_dim = len(self._emb_text("foo"))
        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            dimension=embedding_dim,
            metric_type="IP",  # Inner product distance
            consistency_level="Bounded",  # Strong consistency level
        )

    def load(self, texts: List[str]):
        """
        Load the text data into Milvus.
        """
        data = []
        for i, line in enumerate(tqdm(texts, desc="Creating embeddings")):
            data.append({"id": i, "vector": self._emb_text(line), "text": line})

        self.milvus_client.insert(collection_name=self.collection_name, data=data)

    def retrieve(self, question: str, top_k: int = 3) -> List[str]:
        """
        Retrieve the most similar text data to the given question.
        """
        search_res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[self._emb_text(question)],
            limit=top_k,
            search_params={"metric_type": "IP", "params": {}},  # Inner product distance
            output_fields=["text"],  # Return the text field
        )
        retrieved_texts = [res["entity"]["text"] for res in search_res[0]]
        return retrieved_texts[:top_k]

    def answer(
        self,
        question: str,
        retrieval_top_k: int = 3,
        return_retrieved_text: bool = False,
    ):
        """
        Answer the given question with the retrieved knowledge.
        """
        retrieved_texts = self.retrieve(question, top_k=retrieval_top_k)
        user_prompt = self.USER_PROMPT.format(
            context="\n".join(retrieved_texts), question=question
        )
        response = self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        if not return_retrieved_text:
            return response.choices[0].message.content
        else:
            return response.choices[0].message.content, retrieved_texts

openai_client = OpenAI()
milvus_client = MilvusClient(uri="http://localhost:19530")

my_rag = RAG(openai_client=openai_client, milvus_client=milvus_client)

import os
import urllib.request

url = "https://raw.githubusercontent.com/milvus-io/milvus/master/DEVELOPMENT.md"
file_path = "./Milvus_DEVELOPMENT.md"

if not os.path.exists(file_path):
    urllib.request.urlretrieve(url, file_path)
with open(file_path, "r") as file:
    file_text = file.read()

text_lines = file_text.split("# ")
my_rag.load(text_lines)  # Load the text data into RAG pipeline

question = "what is the hardware requirements specification if I want to build Milvus and run from source code?"
my_rag.answer(question, return_retrieved_text=True)

from ragas import EvaluationDataset
from datasets import Dataset
import pandas as pd

user_input_list = [
    "what is the hardware requirements specification if I want to build Milvus and run from source code?",
    "What is the programming language used to write Knowhere?",
    "What should be ensured before running code coverage?",
]
reference_list = [
    "If you want to build Milvus and run from source code, the recommended hardware requirements specification is:\n\n- 8GB of RAM\n- 50GB of free disk space.",
    "The programming language used to write Knowhere is C++.",
    "Before running code coverage, you should make sure that your code changes are covered by unit tests.",
]
retrieved_contexts_list = []
response_list = []

for user_input in tqdm(user_input_list, desc="Answering questions"):
    response, retrieved_context = my_rag.answer(user_input, return_retrieved_text=True)
    retrieved_contexts_list.append(retrieved_context)
    response_list.append(response)

df = pd.DataFrame(
    {
        "user_input": user_input_list,
        "retrieved_contexts": retrieved_contexts_list,
        "response": response_list,
        "reference": reference_list,
    }
)
rag_results = EvaluationDataset.from_pandas(df)
print(df)

from ragas import evaluate
from ragas.metrics import AnswerRelevancy, Faithfulness, ContextRecall, ContextPrecision

from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
evaluator_llm = LangchainLLMWrapper(llm) 

results = evaluate(
    dataset=rag_results,
    metrics=[
        AnswerRelevancy(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
    ],
)
print(results)
