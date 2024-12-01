
import json
import re
import time
from llm_law_qa.utils import chunks, init_llm, read_jsonl, write_jsonl
import tqdm
import typer
import os
from dotenv import load_dotenv; load_dotenv()
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

def read_questions():
    path = 'data/questions.jsonl'
    questions = read_jsonl(path)

class PromptTemplate:
    # direct answer
    direct = (
        "Bạn là một luật sư am hiểu về luật đất đai.\n"
        "Hãy dùng kiến thức của mình để trả lời câu hỏi sau.\n"
        "Câu hỏi: {}\n"
        "Các lựa chọn:\n"
        "{}\n"
        "Hãy đưa ra câu trả lời theo dạng: 'Đáp án đúng: ...'\n"
    )
    # direct answer with reasoning
    cot = (
        "Bạn là một luật sư am hiểu về luật đất đai.\n"
        "Hãy dùng kiến thức của mình để trả lời câu hỏi sau.\n"
        "Câu hỏi: {}\n"
        "Các lựa chọn:\n"
        "{}\n"
        "Hãy đưa ra lập luận trước khi trả lời."
        "Kết thúc câu trả lời bằng 'Do đó chọn đáp án: ...'\n"
    )

app = typer.Typer()

@app.command()
def direct_answer(input_question_path, answer_save_path, prompt_type='direct'):
    if prompt_type == 'direct':
        prompt_template = PromptTemplate.direct
    else:
        prompt_template = PromptTemplate.cot
    
    model = init_llm()
    questions = read_jsonl(input_question_path)
    start_id = 0
    for i in tqdm.tqdm(range(start_id, len(questions)), total=len(questions) - start_id):
        question = questions[i]
        options = '\n'.join(question['options'])
        prompt = prompt_template.format(question['content'], options)
        answer = model.invoke(prompt)
        answer = {"question_id": i, "answer": answer}
        write_jsonl(answer_save_path, [answer])


def init_emb_model():
    model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
    return model
    
class AnswerRag:
    prompt_template = (
        "Bạn là một luật sư hiểu rõ về luật đất đai."
        "Sau đây là một số văn bản luật để tham khảo\n{}\n"
        "'''\n"
        "Câu hỏi: {}\n"
        "'''\n"
        "Các phương án:\n{}\n"
        "'''\n"
        "Hãy chọn phương án đúng cho câu hỏi trên."
        "Hãy sử dụng thông tin liên quan từ văn bản luật trên để trả lời."
        "Nếu nội dung các điều luật không liên quan đến câu hỏi thì dựa vào kiến thức của bạn để trả lời"
        "Đưa ra câu trả lời dưới dạng: 'Đáp án đúng: ...'\n"
    )
    default_wrong_answer = "0"
    def __init__(self, questions_path, law_path, answer_save_path, model_name='llama3.1', index_name='law', namespace='law'):
        self.questions = read_jsonl(questions_path)
        self.articles = read_jsonl(law_path)
        
        self.model = init_llm(model_name)
        self.emb_model = self.init_emb_model()
        
        self.pc = self.init_store()
        self.index = self.pc.Index(index_name)

        self.index_name = index_name
        self.namespace = namespace
        
        self.doc_limit = 3
        self.answer_pattern = r"Đáp án đúng: ([ABCD])"
        self.answer_save_path = answer_save_path
        
    @staticmethod
    def init_emb_model():
        model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
        return model
    
    def parse_answer(self, answer):
        try:
            answer = re.search(self.answer_pattern, answer).group(1)
        except:
            answer = self.default_wrong_answer
        return answer
    
    def retrieve_relevant_documents(self, question):
        emb = self.emb_model.encode(question).tolist()
        result = self.index.query(
            namespace='law',
            vector=emb,
            top_k=10,
            include_metadata=True,
            include_values=False,
        )
        relevant_docs = []
        for doc in result['matches']:
            article_id = doc['metadata']['article_id']
            relevant_docs.append((article_id, doc['score']))
        relevant_docs = sorted(relevant_docs, key=lambda x: x[1], reverse=True)
        
        # get top k documents
        doc_limit = min(len(relevant_docs), self.doc_limit)
        doc_ids = []
        for doc in relevant_docs:
            if doc[0] not in doc_ids:
                doc_ids.append(doc[0])
                if len(doc_ids) == doc_limit:
                    break
        relevant_docs = doc_ids
        relevant_docs = [self.articles[int(doc-1)]['content'] for doc in relevant_docs]
        return relevant_docs, result
        
    def __call__(self):
        for question in tqdm.tqdm(self.questions):
            relevant_docs, retrieved_result = self.retrieve_relevant_documents(question['content'])
            relevant_docs = '\n'.join(relevant_docs)
            prompt = self.prompt_template.format(
                relevant_docs, question['content'], question['options']
            )
            full_answer = self.model.invoke(prompt)
            answer = self.parse_answer(full_answer)
            data = {
                "question_id": question['id'],
                "answer": answer,
                "full_answer": full_answer
            }
            write_jsonl(self.answer_save_path, [data])
            write_jsonl('tmp/retrieved_result.jsonl', [{"question_id": question['id'], 'retrieved_result': retrieved_result.to_dict()}])
    
    @staticmethod
    def init_store():    
        # connect
        pc = Pinecone(
            api_key=os.getenv("PINECONE")
        )
        return pc   

    @staticmethod
    def insert_document(pc, document_path, embedding_model, index_name='law'):
        # insert an index
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws', 
                    region='us-east-1'
                ) 
            ) 
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        
        # insert document
        with open(document_path, 'r') as f:
            docs = json.load(f)
        doc_content = [doc['content'] for doc in docs]
        doc_embeddings = embedding_model.encode(doc_content)
        
        records = []
        for doc, emb in zip(docs, doc_embeddings):
            records.append({
                "id": str(doc['id']),
                "values": emb,
                "metadata": {
                    "text": doc['content'],
                    "id": doc['id'],
                    "article_id": doc['article_id']
                }
            })
        
        index = pc.Index(index_name)
        for ids_vectors_chunk in chunks(records, batch_size=100):
            index.upsert(vectors=ids_vectors_chunk, namespace='law') 
        
        time.sleep(20)

@app.command()
def answer_rag(questions_path, law_path, answer_save_path):
    model = AnswerRag(questions_path, law_path, answer_save_path)
    model()

@app.command()
def create_db(document_path):
    pc = AnswerRag.init_store()
    embedding_model = init_emb_model()
    AnswerRag.insert_document(pc, document_path, embedding_model)
    
def main():
    pass

if __name__ == '__main__':
    app()