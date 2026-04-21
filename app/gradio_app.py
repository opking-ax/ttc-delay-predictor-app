import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
from gradio.themes import Ocean, Citrus
from src.predict import DelayPredictor
import re

predictor = DelayPredictor()

INCIDENTS = ['General Delay', 'Diversion', 'Operations - Operator', 'Security',
 'Emergency Services', 'Mechanical', 'Investigation', 'Cleaning - Unsanitary',
 'Utilized Off Route', 'Vision', 'Road Blocked - NON-TTC Collision',
 'Collision - TTC', 'Held By', 'Cleaning - Disinfection']

TIME_OF_DAY_CHOICES = ['Overnight', 'AM Peak', 'Midday', 'PM Peak', 'Evening']

DIRECTION_CHOICES = ["N", "S", "E", "W", "B", "U"]

DAY_TUPLE = [
    ("Monday", 0), ("Tuesday", 1), ("Wednesday", 2), ("Thursday", 3), ("Friday", 4), ("Saturday", 5), ("Sunday", 6)
]

MONTH_TUPLE = [
    ("January",   1),  ("February",  2),   ("March",     3),
    ("April",     4),  ("May",        5),  ("June",      6),
    ("July",      7),  ("August",     8),  ("September", 9),
    ("October",  10),  ("November",  11),  ("December", 12),
]


def predict_delay(hour: int, route: str, incident: str, direction: str, day_of_week: int, month: int, time_of_day: str) -> tuple[str, str]:
    raw_data = {
        "hour": hour,
        "route": route,
        "incident": re.sub(r"\s*-\s", "_", incident.lower().strip()).replace(" ", "_"),
        "direction": direction,
        "day_of_week": day_of_week,
        "month": month,
        "time_of_day": time_of_day.lower().strip().replace(" ", "_"),
        "is_weekend": 1 if day_of_week >= 5 else 0,
        "is_am_rush": 1 if 6 <= hour <= 9 else 0,
        "is_pm_rush": 1 if 15 <= hour <= 19 else 0
    }

    result = predictor.predict(raw_data)

    confidence_pct = result['probability'] * 100
    label_text = (
        f"### {result['label']}\n\n"
        f"**Confidence:** {confidence_pct:.1f}%"
    )
    detail_text = (
        f"The model predicts a delay greater than 15 minutes "
        f"with **{confidence_pct:.1f}%** probability for:\n\n"
        f"- Route **{route}** heading **{direction}**\n"
        f"- Incident: **{incident}**\n"
        f"- Hour **{hour:02d}:00** — {time_of_day}"
    )
    return label_text, detail_text


with gr.Blocks(title="TTC Delay Predictor") as demo:
    gr.Markdown("""
    # TTC Bus Delay Predictor
    Predict whether a Toronto Transit Commission bus trip will be delayed by more than 15 minutes.
    Trained on 2022-2024 TTC Open data.
    """)

    gr.Markdown("### Trip Details")
    with gr.Row():
        route_input = gr.Textbox(label="Route Number", value="29", placeholder="e.g., 29", scale=1)
        direction_input = gr.Dropdown(label="Direction", choices=DIRECTION_CHOICES, value="N", scale=1)
        incident_input = gr.Dropdown(label="Incident Type", choices=INCIDENTS, value="Mechanical", scale=3)

    gr.Markdown("### Time Details")
    with gr.Row():
        hour_input = gr.Slider(label="Hour of Day (24h)", minimum=0, maximum=23, step=1, value=8, scale=3)
        tod_input = gr.Dropdown(label="Time of Day", choices=TIME_OF_DAY_CHOICES, value="AM Peak", scale=2)

    with gr.Row():
        day_input = gr.Dropdown(label="Day of Week", choices=DAY_TUPLE, value=2, scale=2)
        month_input = gr.Dropdown(label="Month", choices=MONTH_TUPLE, value=3, scale=2)


    predict_btn = gr.Button("Predict Delay", variant="primary")

    gr.Markdown("### Result")
    with gr.Row():
        label_output  = gr.Markdown(label="Verdict")
        detail_output = gr.Markdown(label="Breakdown")


    predict_btn.click(
        fn=predict_delay,
        inputs=[hour_input, route_input, incident_input, direction_input, day_input, month_input, tod_input],
        outputs=[label_output, detail_output]
    )

    gr.Markdown("---\n*Built with scikit-learn · FastAPI · Gradio · TTC Open Data*")


if __name__ == "__main__":
    demo.launch(server_port=7860, show_error=True, theme=Citrus())
