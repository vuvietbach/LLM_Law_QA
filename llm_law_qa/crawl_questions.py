import json
import random
import time
from typing import Any
import tqdm
from selenium.webdriver.common.by import By
import undetected_chromedriver as uc
import os
import typer

from llm_law_qa.utils import write_jsonl

class BroweserDriver:
    def __init__(self):
        self.driver = uc.Chrome()

    def get(self, url, sleep_time=2):
        # return html content
        self.driver.get(url)
        self.random_scroll()
        return self.driver

    def random_scroll(self,  total_time=5):
        time.sleep(1)
        scroll_by = 400
        for i in range(total_time):
            try:
                direction = random.choice([1])
                # scroll smoothly
                script = f"""
                window.scrollBy({{
                    top: {scroll_by*direction},
                    behavior: 'smooth'
                }});
                """
                self.driver.execute_script(script)
            except:
                pass
            time.sleep(1)


class Selector:
    exam = "#main-content > div.box-wrapper > div.content-wrapper > div.content-header.bg-white.bb-1 > div.bt-1 > div > div > div > ul > li > a"
    question = "#main-content > div.box-wrapper > div.content-wrapper > div.content-body.mb-20 > div > div > div.qas > div"
    question_content = "body > div.page-question.vj-detail-custom > div > div.row.main-qa.d-flex > div.right-qa.col-xs-12.col-sm-8.col-md-8.col-lg-9 > div.bg-white.br-5 > div.title-qa > h1"
    question_option = "body > div.page-question.vj-detail-custom > div > div.row.main-qa.d-flex > div.right-qa.col-xs-12.col-sm-8.col-md-8.col-lg-9 > div.bg-white.br-5 > div.title-qa > div.answer-check.radio label"
    question_answer = "body > div.page-question.vj-detail-custom > div > div.row.main-qa.d-flex > div.right-qa.col-xs-12.col-sm-8.col-md-8.col-lg-9 > div.bg-white.border-simple.answer-container > div > div.question > div > p"
    question_explanation = "body > div.page-question.vj-detail-custom > div > div.row.main-qa.d-flex > div.right-qa.col-xs-12.col-sm-8.col-md-8.col-lg-9 > div.bg-white.border-simple.answer-container > div > div.question > div > div"


class QuestionCrawler:
    base_url = "https://khoahoc.vietjack.com/thi-online/240-cau-trac-nghiem-luat-dat-dai-co-dap-an/105240"
    exam_url_path = 'tmp/exams_urls.json'
    question_url_path = 'tmp/questions_urls.jsonl'
    question_data_path = 'tmp/questions.jsonl'
    
    def __init__(self):
        self.browser = BroweserDriver()
        os.makedirs('tmp', exist_ok=True)
            
        self.crawl_exam_url()
        self.crawl_question_url()
        self.crawl_question_data()
    
    def crawl_exam_url(self):
        driver = self.browser.get(base_url)
        exam_urls = []
        exam_elems = driver.find_elements(By.CSS_SELECTOR, Selector.exam)
        for elem in exam_elems:
            exam_urls.append({
                "exam_id": elem.text,
                "url": elem.get_attribute("href"),
            })
        self.exam_urls = exam_urls
        with open(self.exam_url_path, 'w', encoding='utf-8') as f:
            json.dump(exam_urls, f, ensure_ascii=False, indent=4)
    
    def crawl_question_url(self):
        self.question_urls = []
        for i, exam in tqdm.tqdm(enumerate(self.exam_urls)):
            url = exam['url']
            driver = self.browser.get(url)
            
            q_elems = driver.find_elements(By.CSS_SELECTOR, Selector.question)
            q_metadata = [
                {
                    "exam_id": exam['exam_id'],
                    "question_num": elem.find_element(By.CSS_SELECTOR, "p").text,
                    "url": elem.find_element(By.CSS_SELECTOR, "a").get_attribute("href"),
                    "id": i
                }
                for elem in q_elems
            ]
            self.question_urls.extend(q_metadata)
            write_jsonl(self.question_url_path, q_metadata)
    
    def crawl_question_data(self):
        for i in tqdm.tqdm(range(len(self.question_urls))):
            question = self.question_urls[i]
            url = question['url']
            sleep_time = random.randint(2, 5)
            driver = self.browser.get(url, sleep_time)
            question_data = self.extract_question_data()
            question.update(question_data)
            write_jsonl(self.question_data_path, [question], mode='a')
    
    def extract_question_data(self):
        driver = self.browser.driver
        content = driver.find_element(By.CSS_SELECTOR, Selector.question_content).text
        options = [option.text for option in driver.find_elements(By.CSS_SELECTOR, Selector.question_option)]
        answer = driver.find_element(By.CSS_SELECTOR, Selector.question_answer).text
        try:
            explanation = driver.find_element(By.CSS_SELECTOR, Selector.question_explanation).text
        except:
            explanation = ''
        return {
            "content": content,
            "options": options,
            "answer": answer,
            "explanation": explanation,
        }
        
app = typer.Typer()

@app.command()
def crawl_law_questions():
    QuestionCrawler()

if __name__ == '__main__':
    app()
