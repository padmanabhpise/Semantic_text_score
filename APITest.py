from flask import Flask, request, jsonify
from pickle_solution import preprocess_text,get_bert_embedding,calculate_cosine_similarity,Semantic_similarity_range
app=Flask(__name__)
@app.route('/', methods=['GET'])
def index():
    html="""
    <html>
    <body>
    <form action='/testApi' method='get' >
    <pre>
        text1: <input type=text name=text1 >
        text2: <input type=text name=text2 >
        <input type=submit >
    </pre>
    </form>
    </body>
    </html>
    """
    return html


@app.route('/testApi', methods=['GET'])
def calculate_text_similarity():
    
    text1= request.args.get('text1')
    text2= request.args.get('text2')
    #Preprocess text1 and text2
    
    text1=preprocess_text(text1)
    text2=preprocess_text(text2)
        
    #Claculate BERT embeddings
    embedding1=get_bert_embedding(text1)
    embedding2=get_bert_embedding(text2)
        
    #Calculate coasine similarity
    similarity_score=calculate_cosine_similarity(embedding1,embedding2)
        
    #Convert similarity score to range of 0 to 1
    semantic_similarity=Semantic_similarity_range(similarity_score)
    
    response={'text1':text1,'text2':text2,'similarity score': similarity_score}
    return jsonify(response)
if __name__=='__main__':
    app.run(debug=True)
