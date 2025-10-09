"""自動化評測框架"""
from evaluate import load
import json

class AutoEvaluator:
    def __init__(self, model, test_set):
        self.model = model
        self.test_set = test_set
        self.rouge = load("rouge")
        self.bleu = load("bleu")
    
    def evaluate(self):
        """執行評測"""
        predictions = [self.model.generate(x) for x in self.test_set]
        references = [x["target"] for x in self.test_set]
        
        results = {
            "rouge": self.rouge.compute(predictions=predictions, references=references),
            "bleu": self.bleu.compute(predictions=predictions, references=references),
        }
        return results
    
    def generate_report(self, results, output_path="report.json"):
        """生成評測報告"""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
