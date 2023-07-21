import openai

openai.api_key = "your_api_key"

def generate_css(description):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"{description}\n\nCSS:",
        temperature=0.5,
        max_tokens=100
    )

    css_code = response.choices[0].text.strip()
    return css_code

css = generate_css("Create a button with a gradient from blue to purple, and when hovering over it, it turns green.")
print(css)