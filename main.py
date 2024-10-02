from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from textblob import TextBlob

template = """
Answer the quetsion below.

Here is the conversation history:{context}

Question: {question}

Answer:
"""
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity  # Range from -1 (negative) to 1 (positive)
    return sentiment_score

def handle_conversation():
    context=""
    print("Welcome to Mo AI Chatbot! Type 'exit' to quit.")
    while True:
        try:
            user_input = input("You: ")
            
            if user_input.lower() == "exit":
                print("Goodbye! Have a great day.")
                break

            # Sentiment analysis to adjust tone
            sentiment_score = analyze_sentiment(user_input)
            if sentiment_score < -0.5:
                tone = "empathetic"
            elif sentiment_score > 0.5:
                tone = "enthusiastic"
            else:
                tone = "neutral"
            
            # Passing context and question to the model
            result = chain.invoke({
                "context": context, 
                "question": f" {user_input} (Respond in a {tone} tone.)"
            })

            # Filter out irrelevant or repetitive answers
            if "I'm sorry" in result or result.strip() == "":
                print(f"Bot: Sorry, I didn't quite get that. Could you rephrase?")
            else:
                print(f"Bot: {result}")

            # Updating context for memory
            context += f"\n{user_input}\nAI: {result}"

        except Exception as e:
            print(f"Bot: Oops, something went wrong! Error: {str(e)}")

if __name__=="__main__":
    handle_conversation()