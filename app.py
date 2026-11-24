import gradio as gr
from transformers import pipeline

print("⚡ Loading Qwen AI Model...")
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device="cpu")
print("✅ Model loaded and ready!")

def generate_content(topic, format_type):
    if not topic.strip():
        return "⚠️ Please enter a topic first!"

    # Format-specific instructions so the output actually matches what user asked
    format_rules = {
        "LinkedIn Post": (
            "Write a LinkedIn post with: a strong hook in line 1, "
            "3-4 short paragraphs with blank lines between them, "
            "a one-line takeaway, and 3-5 relevant hashtags at the end."
        ),
        "Twitter/X Thread": (
            "Write a Twitter thread of 5-6 tweets. Each tweet on its own line, "
            "numbered 1/, 2/, 3/, etc. Keep each tweet under 240 characters. "
            "End with a call-to-action tweet."
        ),
        "Email Newsletter": (
            "Write a short email newsletter with: a Subject line, a greeting, "
            "2-3 body paragraphs, a clear call-to-action, and a sign-off."
        ),
        "Blog Outline": (
            "Create a blog outline with: a Title, an Introduction, "
            "4-5 H2 section headings each with 2 bullet points, and a Conclusion."
        ),
    }

    rules = format_rules.get(format_type, format_rules["LinkedIn Post"])

    prompt = (
        f"You are an expert content creator. {rules}\n\n"
        f"Topic: {topic}\n\n"
        f"Write the complete {format_type} now. Finish all sentences. Do not stop early.\n\n"
        f"{format_type}:"
    )

    response = pipe(
        prompt,
        max_new_tokens=600,      # ✅ Was 200 — now the output can finish
        temperature=0.7,
        do_sample=True,
        pad_token_id=pipe.tokenizer.eos_token_id
    )

    full_text = response[0]['generated_text']
    generated_text = full_text.replace(prompt, "").strip()

    # Trim any leftover prompt echo at the start
    if generated_text.startswith(format_type):
        generated_text = generated_text.split(":", 1)[-1].strip()

    return generated_text

with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Qwen AI Content Generator")
    gr.Markdown("Instantly generate professional, ready-to-publish content using the open-source Qwen 2.5 AI model.")

    with gr.Row():
        with gr.Column():
            topic = gr.Textbox(
                label="📝 Your Topic or Idea",
                placeholder="e.g., The importance of continuous learning in the tech industry",
                lines=3,
                value="The importance of continuous learning in the tech industry"
            )
            format_type = gr.Dropdown(
                choices=["LinkedIn Post", "Twitter/X Thread", "Email Newsletter", "Blog Outline"],
                value="LinkedIn Post",
                label="🎯 Content Format"
            )
            btn = gr.Button("✨ Generate Content", variant="primary", size="lg")

        with gr.Column():
            output = gr.Textbox(label="📄 Generated Content", lines=20,
                                placeholder="Your generated content will appear here...")

    gr.Examples(
        examples=[
            ["How AI is transforming healthcare", "LinkedIn Post"],
            ["5 tips for remote work productivity", "Twitter/X Thread"],
            ["Launching our new eco-friendly product line", "Email Newsletter"]
        ],
        inputs=[topic, format_type]
    )

    btn.click(fn=generate_content, inputs=[topic, format_type], outputs=output)

if __name__ == "__main__":
    demo.launch(share=True)