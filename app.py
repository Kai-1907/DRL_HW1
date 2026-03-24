from flask import Flask, render_template, request, jsonify
from logic import run_value_iteration, run_policy_evaluation

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate', methods=['POST'])
def calculate_value_iteration():
    data = request.json
    v_matrix, policy = run_value_iteration(
        data['n'], tuple(data['start']),tuple(data['end']), [tuple(o) for o in data['obstacles']]
    )
    return jsonify({'v_matrix': v_matrix, 'policy': policy})

@app.route('/evaluate', methods=['POST'])
def evaluate_random_policy():
    data = request.json
    # 呼叫你在 logic.py 寫好的隨機策略評估函數
    v_matrix, policy = run_policy_evaluation(
        data['n'], tuple(data['end']), [tuple(o) for o in data['obstacles']]
    )
    return jsonify({'v_matrix': v_matrix, 'policy': policy})

if __name__ == '__main__':
    app.run(debug=True)