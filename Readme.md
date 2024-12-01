# About
This project investigates the performance of large language model in answering multiple choice questions about **Law Property Law** in Vietnam. The questions are crawled from [this site](https://khoahoc.vietjack.com/thi-online/240-cau-trac-nghiem-luat-dat-dai-co-dap-an/105240)
# Setup
## Setup environment
```bash
# set up environment
bash scripts/setup.sh
```
- Follow [this guide](https://docs.pinecone.io/guides/get-started/quickstart) to set up vector database Pinecone for RAG experiment
## Preprocess dataset
```bash
# crawl questions
python llm_law_qa/crawl_questions.py
python llm_law_qa/preprocess_raw_data.py process-questions data/questions.jsonl
# extract law data from pdf
python llm_law_qa/preprocess_raw_data.py  process-law

```
# Experiments:
The performance of LLM are evaluated across 3 settings.
- Output answer directly
- Answer with Chain of thought prompting
- Answer with Retrieval-augmented generation


The expected result is that Retrieval-Augmented Generation (RAG) will enable the LLM to provide the most accurate answers, while the additional reasoning offered by Chain of Thought (CoT) prompting is expected to improve performance compared to answering directly.


Large language models from the LLama 3 family are utilized due to their open-source nature and relatively strong performance. For the RAG setting, the [vietnamese-bi-encoder](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder) sentence transformer is employed.

For the RAG setting, we divide each article into individual terms, as the information within each term is relatively independent. See example below. For query, we use the whole question.
## Chunking
| Article     | RAG chunks   |
|-------------|--------------|
|Điều 2. Đối tượng áp dụng\n1. Cơ quan nhà nước thực hiện quyền hạn và trách nhiệm đại diện chủ sở hữu toàn dân về đất đai, thực hiện nhiệm vụ thống nhất quản lý nhà nước về đất đai.\n2. Người sử dụng đất.\n3. Các đối tượng khác có liên quan đến việc quản lý, sử dụng đất đai.          |- "Điều 2. Đối tượng áp dụng"<br> - "Cơ quan nhà nước thực hiện quyền hạn và trách nhiệm đại diện chủ sở hữu toàn dân về đất đai, thực hiện nhiệm vụ thống nhất quản lý nhà nước về đất đai."<br> - "Các đối tượng khác có liên quan đến việc quản lý, sử dụng đất đai."  |
## Example question:
Luật đất đai chỉ điều chỉnh các quan hệ xã hội có liên quan đến đất đai.

# Result

| Setting     | LLM model    | Accuracy    |
|-------------|--------------|-------------|
| Direct      | LLama3.2 3b  | 80%         |
| COT         | LLama3.2 3b  | 60%         |
| RAG         | LLama3.1 8b  | 50%         |

Contrary to our expectations, the performance of the model when directly answering questions is subpar. The inferior performance of RAG (Retriever-Augmented Generation) can be attributed to the retriever's failure to retrieve relevant documents. The following example illustrates this issue: the query asks about the scope of the law, specifically referring to Article 1, but the retriever fails to retrieve this article.

Query: "Luật đất đai chỉ điều chỉnh các quan hệ xã hội có liên quan đến đất đai."

Retrieved results:
- Điều 20. Nội dung quản lý nhà nước về đất đai
- Điều 223. Các thủ tục hành chính về đất đai
- Điều 11. Hành vi bị nghiêm cấm trong lĩnh vực đất đai
- Phạm vi quản lý đất đai trên đất liền được xác định theo đường địa giới đơn vị hành chính của từng đơn vị hành chính theo quy định của pháp luật.


# Future direction:
- Experiment with meta prompting and use an LLM to extract key question keywords.
- Explore the use of a more advanced embedding model.(OpenAI)
- Experiment with a Vietnamese-specific fine-tuned version of LLaMA.
