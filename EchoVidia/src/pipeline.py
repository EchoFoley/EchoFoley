import json
import os
import time

from action_pool import ActionRegistry
from controller import VideoLLMController
from reasoning_actions import register_reasoning_actions
from sound_design_actions import register_sound_design_actions
from synthesis_actions import register_synthesis_actions


def run(video_path: str, user_instruction: str, output_dir: str) -> str:
    start_time_s = time.time()
    os.makedirs(output_dir, exist_ok=True)

    registry = ActionRegistry()
    register_reasoning_actions(registry)
    register_sound_design_actions(registry)
    register_synthesis_actions(registry)

    controller = VideoLLMController(registry=registry, max_sound_design_rounds=2)
    final_video_path = controller.run(
        video_path=video_path,
        user_instruction=user_instruction,
        output_dir=output_dir,
    )

    time_record = {
        "total": time.time() - start_time_s,
    }
    with open(os.path.join(output_dir, "time_record.json"), "w") as f:
        json.dump(time_record, f, indent=4)

    return final_video_path


if __name__ == "__main__":
    
    video_paths = [
        "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data/multifoley_1.mp4",
        "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data/multifoley_2.mp4",
        "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data/hard_video_1.mp4",
        "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data/muted/long_video_1.mp4",
        "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data/muted/long_video_2.mp4",
        "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data/muted/long_video_3.mp4",
        "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data/muted/long_video_5.mp4",
        "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data/muted/test_video_1.mp4",
        "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data/muted/test_video_2.mp4",
        "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data/muted/test_video_3.mp4",
        "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data/muted/test_video_4.mp4",
        "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data/muted/test_video_5.mp4",
        "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data/muted/test_video_6.mp4",
        "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data/muted/test_video_7.mp4",
        "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data/muted/test_video_8.mp4",
        "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data/muted/test_video_9.mp4",
        "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data/muted/audiogen_omni_1.mp4",
        "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data/muted/audiogen_omni_2.mp4",
    ]

    output_dir_base = "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/results/0811/generation/take_2"
    user_instruction = "make the second event louder and change 'meow' to a lion roar"
    for video_path in video_paths:
        try:
            output_dir = os.path.join(output_dir_base, os.path.basename(video_path).split(".")[0])
            final_video_path = run(video_path, user_instruction, output_dir)
            print(f"Final video path: {final_video_path}")
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
        
        time.sleep(25)