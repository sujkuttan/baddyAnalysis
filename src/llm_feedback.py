import json
import os


SYSTEM_PROMPT = (
    "You are an expert badminton coach for advanced sub-junior national players. "
    "Given structured match analytics, produce concise, encouraging, actionable feedback "
    "covering stroke quality, footwork, court coverage, and fatigue. Do not invent numbers."
)


def build_report_prompt(metrics):
    return (
        "Match analytics (JSON):\n"
        + json.dumps(metrics, indent=2)
        + "\n\nProvide: 1) 3 strengths, 2) 3 improvement areas with specific drills, "
        + "3) fatigue management note. Keep under 250 words."
    )


def generate_feedback(metrics, provider="gemini", api_key=None, model="gemini-1.5-flash"):
    prompt = build_report_prompt(metrics)
    if provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        resp = genai.GenerativeModel(model).generate_content([SYSTEM_PROMPT, prompt])
        return resp.text
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    else:
        return prompt


def write_report(metrics, text, out_path="data/coaching_report.md"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Badminton Coaching Report\n\n")
        f.write(text + "\n\n## Raw Metrics\n\n")
        f.write("```json\n" + json.dumps(metrics, indent=2) + "\n```\n")
    return out_path
