# flask API
from flask import Flask, jsonify, request

from api_helper import * 

app = Flask(__name__)

@app.route('/seq2seq', methods=['POST'])
def seq_question():
    context = request.form['context']
    answer = request.form['answer']
    index = context.find(answer)
    if index != -1:
        answer_start = index
    else:
        answer_start = 0
    # answer_start = request.form['answer_start']
    # context = data['context']
    # answer = data['answer']
    # answer_start = data['answer_start']
    question = ''

    df = pd.DataFrame({'context':[context], 'question':[question], 'answer':[answer], 'answer_start':[answer_start]})
    dataframe = build_lexical_features(df)
    csv_path = 'test.csv'
    dataframe.to_csv(csv_path, index=False)
    test_dataset = TabularDataset(path=csv_path, format='csv', fields=fields, skip_header=True)
    # Display prediction
    for example in test_dataset.examples:
        src = vars(example)['context']
        ans = vars(example)['bio']
        lex = vars(example)['lex']

        # print('context: ', ' '.join(src))
        question, logits = predict_question(model, src, ans, lex)
        # print('predicted: ', " ".join(question))
        # print()
    return jsonify({'question': " ".join(question)})



if __name__ == '__main__':
    app.run(debug=True, port=5005)