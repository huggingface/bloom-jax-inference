import gradio as gr
import requests
import os
from screenshot import BG_COMP, BOX_COMP, GENERATION_VAR, PROMPT_VAR, main
from pathlib import Path

title = "JAX / Flax BLOOM 🌸"

description = """Gradio Demo for BLOOM. To use it, simply add your text, or click one of the examples to load them.
Tips:
- Do NOT talk to BLOOM as an entity, it's not a chatbot but a webpage/blog/article completion model.
- For the best results: MIMIC a few sentences of a webpage similar to the content you want to generate.
Start a paragraph as if YOU were writing a blog, webpage, math post, coding article and BLOOM will generate a coherent follow-up. Longer prompts usually give more interesting results.

Options:
- sampling: imaginative completions (may be not super accurate e.g. math/history)
- greedy: accurate completions (may be more boring or have repetitions)
"""

wip_description = """Gradio Demo for JAX / Flax BLOOM. The 176B BLOOM model running on a TPU v3-256 pod, with 2D model parallelism and custom mesh axes.


Note: rendering of the screenshot is currently not optimised. To experience the true speed of JAX / Flax, tick 'just output raw text'.

Tips:
- Do NOT talk to BLOOM as an entity, it's not a chatbot but a webpage/blog/article completion model.
- For the best results: MIMIC a few sentences of a webpage similar to the content you want to generate.
Start a paragraph as if YOU were writing a blog, webpage, math post, coding article and BLOOM will generate a coherent follow-up. Longer prompts usually give more interesting results.

Options:
- sampling: imaginative completions (may be not super accurate e.g. math/history)
- greedy: accurate completions (may be more boring or have repetitions)
"""

API_URL = os.getenv("API_URL")

examples = [

    ['To do a "farduddle" means to jump up and down really fast. An example of a sentence that uses the word farduddle is:',
        64, "sampling", True],
    ['A poem about the beauty of science by Alfred Edgar Brittle\nTitle: The Magic Craft\nIn the old times', 64,
     "sampling", True],
    ['استخراج العدد العاملي في لغة بايثون:', 64, "greedy", True],
    ["Pour déguster un ortolan, il faut tout d'abord", 64, "sampling", True],
    ['Traduce español de España a español de Argentina\nEl coche es rojo - el auto es rojo\nEl ordenador es nuevo - la computadora es nueva\nel boligrafo es negro -',
        64, "sampling", True],
    ['Estos ejemplos quitan vocales de las palabras\nEjemplos:\nhola - hl\nmanzana - mnzn\npapas - pps\nalacran - lcrn\npapa -',
        64, "sampling", True],
    ["Question: If I put cheese into the fridge, will it melt?\nAnswer:", 64, "greedy", True],
    ["Math exercise - answers:\n34+10=44\n54+20=", 64, "sampling", True],
    ["Python - code to compute the factorial of a number:", 64, "greedy", True],
    ["Question: Where does the Greek Goddess Persephone spend half of the year when she is not with her mother?\nAnswer:",
        64, "sampling", True],
    ["spelling test answers.\nWhat are the letters in « language »?\nAnswer: l-a-n-g-u-a-g-e\nWhat are the letters in « Romanian »?\nAnswer:",
        64, "sampling", True],
    ['A "whatpu" is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is:',
     64, "sampling", True],
]


def query(payload):
    print(payload)
    response = requests.post(API_URL, json=payload)
    print(response)
    return response.json()


def inference(input_sentence, max_length, sample_or_greedy, raw_text=True):
    do_sample = sample_or_greedy == "sampling"
    payload = {
        "inputs": input_sentence,
        "do_sample": do_sample,
        # "max_new_tokens": max_length
    }

    data = query(
        payload
    )

    if raw_text:
        return None, data[0]['generated_text'][0]

    width, height = 3326, 3326
    assets_path = "assets"
    font_mapping = {
        "latin characters (faster)": "DejaVuSans.ttf",
        "complete alphabet (slower)": "GoNotoCurrent.ttf"
    }
    working_dir = Path(__file__).parent.resolve()
    font_path = str(working_dir / font_mapping["complete alphabet (slower)"])
    img_save_path = str(working_dir / "output.jpeg")
    colors = {
        BG_COMP: "#000000",
        PROMPT_VAR: "#FFFFFF",
        GENERATION_VAR: "#FF57A0",
        BOX_COMP: "#120F25",
    }

    # TODO: fix screenshot
    new_string = data[0]['generated_text'][0].split(input_sentence, 1)[1]

    _, img = main(
        input_sentence,
        new_string,
        width,
        height,
        assets_path=assets_path,
        font_path=font_path,
        colors=colors,
        frame_to_box_margin=200,
        text_to_text_box_margin=50,
        init_font_size=142,
        right_align=False,
    )
    return img, data[0]['generated_text'][0]


gr.Interface(
    inference,
    [
        gr.inputs.Textbox(label="Input"),
        gr.inputs.Radio([64], default=64, label="Tokens to generate"),
        gr.inputs.Radio(["sampling", "greedy"], label="Sample or greedy", default="sampling"),
        gr.Checkbox(label="Just output raw text", value=True),
    ],
    ["image", "text"],
    examples=examples,
    # article=article,
    cache_examples=False,
    title=title,
    description=wip_description
).launch()
