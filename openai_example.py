from openai import OpenAI

client = OpenAI()

# models = client.models.list()
# for model in models:
#     print(model.id)


response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {"role": "user", "content": "Write a Python function to reverse a string."}
    ],
    max_completion_tokens=1000
)

print(response.choices[0].message.content)
