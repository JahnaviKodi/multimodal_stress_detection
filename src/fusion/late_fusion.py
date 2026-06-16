import numpy as np

def late_fusion(face_score, audio_score, physio_score, w_face=0.33, w_audio=0.33, w_physio=0.34):
    final_score = (w_face * face_score) + (w_audio * audio_score) + (w_physio * physio_score)

    if final_score >= 0.5:
        return "stress", final_score
    else:
        return "non_stress", final_score


if __name__ == "__main__":
    # Example predictions from the three models
    face_score = 1       # stress
    audio_score = 0      # non-stress
    physio_score = 1     # stress

    label, score = late_fusion(face_score, audio_score, physio_score)

    print("Final fused prediction:", label)
    print("Fusion score:", round(score, 3))