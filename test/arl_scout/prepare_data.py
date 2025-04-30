import pandas as pd
from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO


import os

def tag_important_segments(segments):
    action_keywords = ["go", "turn", "face", "take a picture", "move", "rotate"]
    transition_keywords = ["room", "door", "doorway", "hallway", "center"]
    clarification_markers = ["which", "what", "do you mean", "can you repeat", "i had trouble"]
    
    for seg in segments:
        score = 0
        segment_texts = [s['text'].lower() for s in seg['segment']]
        roles = [s['role'] for s in seg['segment']]
        cmd_turns = [s['text'].lower() for s in seg['segment'] if s['role'] == "CMD"]

        # 1. Commander instructions
        commander_instr = sum(
            any(kw in text for kw in action_keywords)
            for text in cmd_turns
        )
        score += commander_instr * 1.0  # +1 per instruction

        # 2. Clarification patterns (back-and-forth between DM and CMD with clarification language)
        clarification_turns = 0
        for i in range(1, len(seg['segment'])):
            curr, prev = seg['segment'][i], seg['segment'][i - 1]
            if (
                curr['role'].startswith("DM") and
                prev['role'] == "CMD" and
                any(mark in curr['text'].lower() for mark in clarification_markers)
            ):
                clarification_turns += 1
        score += clarification_turns * 1.5

        # 3. Transitions — location-changing language
        transitions = sum(
            any(tk in text for tk in transition_keywords)
            for text in segment_texts
        )
        score += transitions * 0.75

        # 4. Long segment
        score += 0.1 * len(seg['segment'])  # +0.1 per utterance

        seg['importance_score'] = round(score, 3)
        seg['commander_instructions'] = commander_instr
        seg['clarifications'] = clarification_turns
        seg['transitions'] = transitions
        seg['num_turns'] = len(seg['segment'])

    return segments

TICKS = [
    (0, 0, "TV"),
    (28, 28, "Dark Room"),
    (48, 48, "Pitch Black"),
    (58, 58, "Door"),
    (78, 78, "bright room w/ shovel"),
    (122, 122, "turn to door"),
    (131, 161, "static at door"),
    (166, 166, "rapid turn to TV room"),
    (200, 200, "Moved closer to TV"),
    (202, 245, "static"),
    (245, 266, "Moved closer to TV, static"),
    (266, 289, "static TV"),
    (289, 298, "turn to wall/poster"),
    (305, 305, "full turn to poster"),
    (357, 357, "turn away from poster"),
    (375, 375, "face hallway"),
    (411, 426, "move into dark room"),
    (445, 445, "face dark room window"),
    (471, 471, "move to door"),
    (503, 503, "move in dark room"),
    (529, 529, "turn to lit area"),
    (638, 638, "turn & move to new area"),
    (696, 696, "big move to lit room"),
    (725, 725, "slight turn"),
    (767, 767, "move to calendar"),
    (849, 871, "massive movement"),
    (933, 933, "move to water jug"),
    (955, 955, "turn to cubes"),
    (1000, 1000, "turn to water jug"),
    (1020, 1020, "move to water jug"),
    (1031, 1031, "switch angle"),
]


def prepare_frames_for_model(folder):
    files = os.listdir(folder)
    

def generate_plot(up_to_idx, data, agent_response):
    fig, ax = plt.subplots(figsize=(5, 4))
    times = [d["time"] for d in data[:up_to_idx+1]]
    informative_scores = [d["informative_score"] for d in data[:up_to_idx+1] if d.get("role", "user") != "assistant"]
    relevance_scores = [d["relevance_score"] for d in data[:up_to_idx+1] if d.get("role", "user") != "assistant"]
    uncertainty_scores = [d["uncertainty_score"] for d in data[:up_to_idx+1] if d.get("role", "user") != "assistant"]

    ax.plot(times, informative_scores, label="Informative Score")
    ax.plot(times, relevance_scores, label="Relevance Score")
    ax.plot(times, uncertainty_scores, label="Uncertainty Score")
    ax.set_xlim(0, data[-1]["time"])  # Fix x-axis to full time span
    ax.set_ylim(0, 1)  # Fix y-axis from 0 to 1
    ax.set_xlabel("Time")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True)

    if agent_response:
        ax.text(
            0.01, 0.95,  # x, y position (axes fraction)
            f"Agent: {agent_response}",
            transform=ax.transAxes,  # important: use axes coordinates, not data
            fontsize=8,
            va='top', ha='left',
            wrap=True,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def load_frame_segments(filepath):
    df = pd.read_excel(filepath)
    df.columns = [col.strip() for col in df.columns]

    frame_segments = []
    last_frame_idx = -1

    for idx, row in df.iterrows():
        if pd.notna(row['Image Stream']):
            frame_name = row['Image Stream']
            frame_time = row['Timestamp']

            # Capture everything between the last frame and this one
            segment_rows = df.iloc[last_frame_idx + 1 : idx]

            segment = {
                "frame": frame_name,
                "timestamp": frame_time,
                "segment": []
            }

            for _, seg_row in segment_rows.iterrows():
                for role_col, role_tag in [
                    ('Commander', 'CMD'),
                    ('DM->CMD', 'DM->CMD'),
                    ('DM->RN', 'DM->RN'),
                    ('RN', 'RN'),
                ]:
                    if pd.notna(seg_row.get(role_col)):
                        segment["segment"].append({
                            "timestamp": seg_row['Timestamp'],
                            "role": role_tag,
                            "text": seg_row[role_col]
                        })

            frame_segments.append(segment)
            last_frame_idx = idx  # update boundary

    return frame_segments


if __name__ == "__main__":

    samples = load_frame_segments("datasets/ARL-SCOUT/data/iSCOUT/p1.01_main1_iscout.xlsx")
    tagged_segments = tag_important_segments(samples)
    top = sorted(tagged_segments, key=lambda x: x['importance_score'], reverse=True)

    pprint(top[:3])
    # pprint(samples)
