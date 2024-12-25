from flask import Flask, request, jsonify
from openai import OpenAI
import os

# Đọc API key từ file
with open("./ieltsGPT/key.txt", 'r', encoding='utf-8') as f:
    key = [i.strip() for i in f.readlines()]
client = OpenAI(api_key=key[0])

app = Flask(__name__)

# Hàm để gọi LLM (OpenAI API)
def ask_llm(prompt):
    resp = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
    answer = resp.choices[0].message.content
    return answer.strip()

# Hàm tải prompt cho IELTS
def load_ielts_prompt(task=1, aspect=None):
    if not aspect:
        aspects = ['ta', 'cc', 'lr', 'gra']
    else:
        aspects = [aspect]
    
    text = []
    for aspect in aspects:
        with open(f"ieltsGPT/prompts/ielts/ielts_writing_{task}_prompt_{aspect}.txt", 'r', encoding='utf-8') as f:
            text.append((aspect, f.read().strip()))
    return text

# Hàm tải prompt cho TOEFL
def load_toefl_prompt(task=1):
    if task == 1:  # integrated_writing
        text = open("ieltsGPT/prompts/toefl/toefl_integrated_writing.txt", 'r', encoding='utf-8').read().strip()
    elif task == 2:  # academic_discussion
        text = open("ieltsGPT/prompts/toefl/toefl_academic_discussion.txt", 'r', encoding='utf-8').read().strip()
    return text

# Hàm làm sạch văn bản
def clean(text):
    return text.replace('<br>', "\n")

# Hàm đánh giá bài viết IELTS
@app.route('/evaluateIELTS', methods=['POST'])
def evaluate_ielts():
    data = request.get_json()

    if 'exam' not in data or data['exam'] != 'IELTS':
        return jsonify({'error': 'Invalid exam type'}), 400

    task = data.get('task', 1)
    essay = data.get('essay', '')
    reading_prompt = data.get('reading_prompt', '')
    
    if not essay or not reading_prompt:
        return jsonify({'error': 'Essay and reading prompt are required'}), 400

    prompts = load_ielts_prompt(task)
    text = essay
    feedback = {}

    done = []
    for aspect, prompt in prompts:
        query = prompt + "\nReading Prompt: " + reading_prompt + "\nEssay: " + text
        try:
            feedback[aspect] = clean(ask_llm(query))
            done.append(aspect)
        except Exception as e:
            feedback[aspect] = f"Error: {str(e)}"

    return jsonify({
        'exam': 'IELTS',
        'task': task,
        'feedback': feedback
    })

# Hàm đánh giá bài viết TOEFL
@app.route('/evaluateTOEFL', methods=['POST'])
def evaluate_toefl():
    data = request.get_json()

    if 'exam' not in data or data['exam'] != 'TOEFL':
        return jsonify({'error': 'Invalid exam type'}), 400

    task = data.get('task', 1)
    essay = data.get('essay', '')
    reading_prompt = data.get('reading_prompt', '')
    
    if not essay or not reading_prompt:
        return jsonify({'error': 'Essay and reading prompt are required'}), 400

    prompt = load_toefl_prompt(task)
    text = essay
    query = prompt + "\nReading Prompt: " + reading_prompt + "\nEssay: " + text
    feedback = {}

    try:
        feedback['overall'] = clean(ask_llm(query))
    except Exception as e:
        feedback['overall'] = f"Error: {str(e)}"

    return jsonify({
        'exam': 'TOEFL',
        'task': task,
        'feedback': feedback
    })

if __name__ == '__main__':
    app.run(debug=True)
