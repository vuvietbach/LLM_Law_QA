import itertools
import json
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from paddleocr import PaddleOCR
from PIL import Image

class VietOCR:
    def __init__(self) -> None:
        # initialize detection model
        self.detector = PaddleOCR(use_angle_cls=False, lang="vi")
        # initialize recognition model
        config = Cfg.load_config_from_name('vgg_transformer')
        config['cnn']['pretrained'] = True
        config['predictor']['beamsearch'] = True
        config['device'] = 'gpu'
        self.recognition_model = Predictor(config)
        
    def __call__(self, image):
        # Text detection
        boxes = self.detector.ocr(image, cls=False, det=True, rec=False)
        boxes = boxes[:][:][0]

        results = []
        padding = 0
        for box in boxes:
            x1, y1, x2, y2 = int(box[0][0])-padding, int(box[0][1])-padding, int(box[2][0])+padding, int(box[2][1])+padding
            results.append({"box": [x1, y1, x2, y2]})
        boxes = boxes[::-1]

        # Text recognization
        for i in range(len(results)):
            box = results[i]["box"]
            cropped_image = image[box[1]:box[3], box[0]:box[2]]
            try:
                cropped_image = Image.fromarray(cropped_image)
            except:
                continue
            rec_result = self.recognition_model.predict(cropped_image)
            results[i]['text'] = rec_result

        return results
    
def write_jsonl(file, lines, mode='a'):
    with open(file, mode, encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
            
def init_llm(model='llama3.2'):
    from langchain_ollama import OllamaLLM
    model = OllamaLLM(model=model)
    return model

def read_jsonl(file):
    with open(file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]

def chunks(iterable, batch_size=200):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))