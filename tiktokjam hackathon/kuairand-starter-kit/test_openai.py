from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.6",
    input=(
        "You are helping test an autonomous machine learning "
        "research agent. Reply with exactly: OPENAI CONNECTED"
    )
)

print(response.output_text)