import gradio as gr
from mlx_lm import load, generate
from huggingface_hub import snapshot_download
import os

def download_model(repo_id):
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    return snapshot_download(repo_id=repo_id, cache_dir=cache_dir)

def load_model(repo_id):
    model_path = download_model(repo_id)
    model, tokenizer = load(model_path)
    return model, tokenizer

def answer(state, state_chatbot, text, model, tokenizer):
    input_text = f"### 질문: {text}\n\n### 답변:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    response = generate(model, tokenizer, prompt=input_text, verbose=True, temp=0.2, max_tokens=256)

    if isinstance(response, list):
        response = response[0]

    if isinstance(response, str):
        msg = response
    else:
        msg = tokenizer.decode(response, skip_special_tokens=True)

    new_state = [{"role": "이전 질문", "content": text}, {"role": "이전 답변", "content": msg}]
    state = state + new_state
    state_chatbot = state_chatbot + [(text, msg)]
    return state, state_chatbot, state_chatbot

with gr.Blocks(css="#chatbot .overflow-y-auto{height:750px}") as demo:
    state = gr.State(
        [
            {"role": "맥락", "content": "Explain the model..."},
            {"role": "명령어", "content": "You are the AI Chatbot Response kindly."}
        ]
    )
    state_chatbot = gr.State([])
    model_state = gr.State(None)
    tokenizer_state = gr.State(None)

    with gr.Row():
        gr.HTML("<h1>mlx ChatBot 💻</h1>")

    with gr.Row():
        model_input = gr.Textbox(label="Enter Hugging Face Repo (e.g., sosoai/Hansoldeco-Gemma-2-9b-it-v0.1-mlx)")
        load_button = gr.Button("Load Model")

    with gr.Row():
        chatbot = gr.Chatbot(elem_id="chatbot")

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Send a message...", container=False)

    def load_selected_model(repo_id):
        model, tokenizer = load_model(repo_id)
        return model, tokenizer, gr.update(interactive=True), f"Model {repo_id} loaded successfully!"

    load_button.click(
        load_selected_model,
        inputs=[model_input],
        outputs=[model_state, tokenizer_state, txt, model_input]
    )

    txt.submit(answer, inputs=[state, state_chatbot, txt, model_state, tokenizer_state], outputs=[state, state_chatbot, chatbot])
    txt.submit(lambda: "", None, txt)

demo.launch(debug=True, server_name="0.0.0.0", share=True)
