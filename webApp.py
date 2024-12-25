from flask import Flask, render_template, request, jsonify
import webbrowser
import os
from flask_cors import CORS
import json

import lambdaTTS
import lambdaSpeechToScore
import lambdaGetSample
from openai import OpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
import getpass
import os

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = '*'

rootPath = ''

@app.route(rootPath+'/')
def landing():
    return render_template('landing.html')

@app.route(rootPath+'/main')
def main():
    return render_template('main.html')
@app.route(rootPath+'/chatbot')
def chatbot():
    return render_template('chatbot.html')
@app.route(rootPath+'/evaluate')
def evaluate():
    return render_template('evaluate.html')

@app.route(rootPath+'/getAudioFromText', methods=['POST'])
def getAudioFromText():
    event = {'body': json.dumps(request.get_json(force=True))}
    return lambdaTTS.lambda_handler(event, [])

@app.route(rootPath+'/getSample', methods=['POST'])
def getNext():
    event = {'body':  json.dumps(request.get_json(force=True))}
    return lambdaGetSample.lambda_handler(event, [])

@app.route(rootPath+'/GetAccuracyFromRecordedAudio', methods=['POST'])
def GetAccuracyFromRecordedAudio():
    event = {'body': json.dumps(request.get_json(force=True))}
    lambda_correct_output = lambdaSpeechToScore.lambda_handler(event, [])
    return lambda_correct_output
@app.route('/askQuestion', methods=['POST'])
def ask_question():
    try:
        
        

        # Load embedder and vector store
        embedder = OpenAIEmbeddings(model="text-embedding-3-small")
        vector = FAISS.load_local('./vector_store_test_case/', embedder, allow_dangerous_deserialization=True)
        retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 1})
        
        # Initialize language model
        llm = ChatOpenAI(model="gpt-4o-mini")

        # Define system prompt
        system_prompt = (
            "Bạn là trợ lý hỗ trợ các câu hỏi liên quan đến học tiếng Anh. "
            "Hãy sử dụng các đoạn thông tin trích xuất được dưới đây để trả lời câu hỏi. "
            "Nếu không trích xuất được thông tin nào, hãy sử dụng chính kiến thức của bạn để trả lời. "
            "Giữ câu trả lời ngắn gọn, dễ hiểu, và đi thẳng vào trọng tâm. "
            "Luôn nhớ rằng khi kết thúc câu trả lời hãy hỏi người dùng có cần thêm thông tin gì không. "
            "\n\n"
            "{context}"
        )

        # Parse user input
        data = request.get_json(force=True)
        user_question = data.get("question", "").strip()
        if not user_question:
            return jsonify({"error": "Question is required."}), 400

        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(user_question)  # Fixed to pass string, not dict
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Generate response
        prompt = system_prompt.format(context=context) + "\n" + user_question
        response = llm(prompt)

        # Return response
        return jsonify({"answer": response.content})

    except Exception as e:
        # Handle errors gracefully
        return jsonify({"error": str(e)}), 500
    
# Đọc API key từ file
with open("./ieltsGPT/key.txt", 'r', encoding='utf-8') as f:
    key = [i.strip() for i in f.readlines()]
client = OpenAI(api_key=key[0])
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

def load_toefl_prompt(task=1):
    if task == 1:  
        text = open("ieltsGPT/prompts/toefl/toefl_integrated_writing.txt", 'r', encoding='utf-8').read().strip()
    elif task == 2:  
        text = open("/ieltsGPT/prompts/toefl/toefl_academic_discussion.txt", 'r', encoding='utf-8').read().strip()
    return text


def clean(text):
    return text.replace('<br>', "\n")

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

if __name__ == "__main__":
    language = 'de'
    print(os.system('pwd'))
    webbrowser.open_new('http://127.0.0.1:3000/')
    app.run(host="0.0.0.0", port=3000)