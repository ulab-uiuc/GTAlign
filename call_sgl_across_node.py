from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:30003/v1"
)

resp = client.chat.completions.create(
    model="default",
    messages=[{"role":"user","content":"Hello over IPv6!"}]
)
print(resp.choices[0].message.content)