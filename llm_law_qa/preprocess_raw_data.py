import re
import typer
from llm_law_qa.utils import read_jsonl, write_jsonl

app = typer.Typer()

@app.command()
def process_questions(input_path, output_path=None):
    # extract correct answer from questions
    answer_pattern = r"Chọn đáp án ([ABCD])"
    questions = read_jsonl(input_path)
    for question in questions:
        question['correct_answer'] = re.search(answer_pattern, question['answer']).group(1)
    if output_path is None:
        output_path = input_path
    write_jsonl(output_path, questions, mode='w')    

@app.command()
def process_law(input_path, output_path=None):
    # currently not support scanned PDF
    # TODO: Add support for scanned PDF
    ProcessLawFromPDF()
    
class ProcessLawFromPDF:
    # Step 1: Extract text blocks
    # Step 2: Remove unrelated text blocks
    # Step 3: Group text blocks into articles
    law_pdf_path = 'data/luatdatdai.pdf'
    save_block_path = 'data/processed/article_term.jsonl'
    save_article_path = 'data/processed/articles.jsonl'
    def __init__(self):
        text_blocks = self.extract_text_blocks(self.law_pdf_path)
        self.group_text_blocks(text_blocks, self.save_block_path, self.save_article_path)
        
    @staticmethod
    def extract_text_blocks(pdf_path):
        # Step 1 + 2
        import fitz
        doc = fitz.open(pdf_path)
        text_blocks = []
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                try:
                    content = []
                    for line in block["lines"]:
                        for span in line["spans"]:
                            content.append(span["text"])
                    content = " ".join(content)
                    # Remove unrelated text blocks
                    if 'about' in content:
                        continue
                except:
                    continue
        return text_blocks
    
    @staticmethod
    def group_text_blocks(text_blocks, save_block_path, save_article_path):
        ########################################
        # Overall:
        # Step 3
        # Remove unrelated text blocks at the end
        ########################################
        
        # Group text blocks into articles
        import re
        articles = []
        next_article_id = 1
        current_start_block = -1
        pattern = r'Điều ([\d]+)'
        for i, block in enumerate(text_blocks):
            match = re.search(pattern, block['content'])
            if match and match.group(1) == str(next_article_id):
                if next_article_id > 1:
                    articles.append({"id": next_article_id - 1, "blocks": text_blocks[current_start_block:i]})
                next_article_id += 1
                current_start_block = i
        articles.append({"id": next_article_id - 1, "lines": text_blocks[current_start_block:]})
        
        # Remove unrelated text blocks at the end
        pattern = '(Chương|Mục)'
        for article in articles:
            end_block = -1
            for i, block in enumerate(article['blocks']):
                if re.match(pattern, block['content']):
                    end_block = i
                    break
            if end_block != -1:
                article['blocks'] = article['blocks'][:end_block]
    
        # save each blocks
        blocks = []
        current_block = 0
        for article in articles:
            for block in article['blocks']:
                blocks.append({
                    "id": current_block,
                    "content": block['content'],
                    "article_id": article['id'],
                    "page": block['page']
                })
                current_block += 1
        write_jsonl(save_block_path, blocks, mode='w')
        
        # save each article
        new_articles = []
        for article in articles:
            new_articles.append({
                "id": article['id'],
                "content": '\n'.join([block['content'] for block in article['blocks']])
            })
        write_jsonl(save_article_path, new_articles, mode='w')

if __name__ == "__main__":
    app()