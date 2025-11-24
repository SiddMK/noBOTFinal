from timetable_retriever import get_timetable_answer
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")  # or any model you use

def answer_timetable_query(user_query):
    # Step 1: Retrieve relevant data
    retrieved_docs = get_timetable_answer(user_query)

    # Step 2: Combine retrieved data and query
    context = "\n".join(retrieved_docs)
    prompt = f"Answer the question using the timetable info below:\n\n{context}\n\nQuestion: {user_query}"

    # Step 3: Generate a natural answer
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    return response.output_text

