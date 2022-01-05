from flask import Flask, request, jsonify

from main import chatWithBot

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    return "Hello World"


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    chatInput = request.args.get('message')
    return jsonify(chatBotReply=chatWithBot(chatInput))


if __name__ == '__main__':
    app.debug = True
    app.run()
