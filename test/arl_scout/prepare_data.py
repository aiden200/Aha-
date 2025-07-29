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

def prepare_frames_for_model(folder):
    files = os.listdir(folder)
    

def generate_plot(up_to_idx, data, agent_response):
    fig, ax = plt.subplots(figsize=(5, 4))
    times = [d["time"] for d in data[:up_to_idx+1]]
    informative_scores = [d["informative_score"] for d in data[:up_to_idx+1] if d.get("role", "user") != "assistant"]
    relevance_scores = [d["relevance_score"] for d in data[:up_to_idx+1] if d.get("role", "user") != "assistant"]
    uncertainty_scores = [d["uncertainty_score"] for d in data[:up_to_idx+1] if d.get("role", "user") != "assistant"]

    # ax.plot(times, informative_scores, label="Informative Score")
    fig.patch.set_facecolor('#231f20')
    ax.set_facecolor('#231f20')
    ax.plot(times, relevance_scores, label="Relevance Score", color="BLACK", linewidth=3)
    ax.plot(times, informative_scores, label="Informativeness scores Score", color="BLUE", linewidth=3)


    # ax.plot(times, uncertainty_scores, label="Uncertainty Score")
    ax.set_xlim(0, data[-1]["time"])  # Fix x-axis to full time span
    ax.set_ylim(0, 0.7)  # Fix y-axis from 0 to 1
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Score", color="white")
    # ax.legend()
    ax.grid(True)

    # if agent_response:
    #     ax.text(
    #         0.01, 0.95,  # x, y position (axes fraction)
    #         f"Agent: {agent_response}",
    #         transform=ax.transAxes,  # important: use axes coordinates, not data
    #         fontsize=8,
    #         va='top', ha='left',
    #         wrap=True,
    #         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    #     )

    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    leg = ax.legend(
        facecolor='black',   # legend box fill
        edgecolor='white',   # legend box border
    )
    for text in leg.get_texts():
        text.set_color('white')
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
