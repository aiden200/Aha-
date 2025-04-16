import pandas as pd
from pprint import pprint


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


samples = load_frame_segments("datasets/ARL-SCOUT/data/iSCOUT/p1.01_main1_iscout.xlsx")
tagged_segments = tag_important_segments(samples)
top = sorted(tagged_segments, key=lambda x: x['importance_score'], reverse=True)

pprint(top[:3])
# pprint(samples)
