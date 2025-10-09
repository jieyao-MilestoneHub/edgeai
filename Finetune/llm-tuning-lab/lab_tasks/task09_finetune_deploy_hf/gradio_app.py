"""Gradio 互動介面"""
import gradio as gr
from transformers import pipeline
import argparse

def create_interface(model_name):
    # 載入模型
    generator = pipeline("text-generation", model=model_name)
    
    def generate_text(prompt, max_length, temperature):
        outputs = generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
        )
        return outputs[0]['generated_text']
    
    # 建立介面
    interface = gr.Interface(
        fn=generate_text,
        inputs=[
            gr.Textbox(label="輸入你的問題", lines=3),
            gr.Slider(50, 200, value=100, label="最大長度"),
            gr.Slider(0.1, 2.0, value=0.7, label="Temperature"),
        ],
        outputs=gr.Textbox(label="AI 回應", lines=5),
        title="🤖 我的 AI 助手",
        description="由微調模型驅動的智能助手",
        examples=[
            ["如何學習 Python？", 100, 0.7],
            ["推薦一個編輯器", 100, 0.7],
        ],
    )
    
    return interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="模型名稱")
    parser.add_argument("--share", action="store_true", help="產生公開連結")
    args = parser.parse_args()
    
    interface = create_interface(args.model_name)
    interface.launch(share=args.share)
