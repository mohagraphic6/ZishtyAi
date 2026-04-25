from assistant.brain import chat, reset

print("Zishty AI - Your Life Assistant")
print("Type 'reset' to start over, 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() == "quit":
        print("Zishty: Take care. I'm always here when you need me.")
        break
    if user_input.lower() == "reset":
        reset()
        print("Zishty: Fresh start. What's on your mind?")
        continue
    reply = chat(user_input)
    print(f"Zishty: {reply}\n")
