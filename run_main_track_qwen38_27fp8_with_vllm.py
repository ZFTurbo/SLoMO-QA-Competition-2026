import os
import glob
import json
import warnings
import sys
import pandas as pd
from decord import VideoReader, cpu
from openai import OpenAI

warnings.filterwarnings("ignore")

MODEL_NAME = "qwen38-27bfp8"
TYPE = 'private'
# TYPE = 'public'
INPUT_PATH = sys.argv[1] + '/'


def proc_with_qwen():
    questions = pd.read_csv(INPUT_PATH + 'sf20k_{}_test_questions.csv'.format(TYPE))
    files = glob.glob(INPUT_PATH + 'sf20k_{}_test_videos/*.*'.format(TYPE))
    print("Files to proc:", len(files))

    client = OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY"
    )

    # Must be the same as --served-model-name in vLLM
    model_name = "qwen38-27bfp8"
    N_FRAMES = 200

    out_cache_folder = './cache/{}_data/'.format(model_name)
    os.makedirs(out_cache_folder, exist_ok=True)

    out_file = './sf20k_{}_test_questions_results_{}_{}.json'.format(TYPE, N_FRAMES, model_name)
    results = []

    for f in files:
        video_id = os.path.basename(f)[:-4]
        print(video_id)

        part = questions[questions['video_id'] == video_id]

        vr = VideoReader(f, ctx=cpu(0))
        total_frames = len(vr)
        original_fps = vr.get_avg_fps()
        video_duration = total_frames / original_fps

        print(f"Total frames in file: {total_frames}")
        print(f"Length of video: {video_duration:.2f} sec, FPS: {original_fps:.2f}")

        target_fps = N_FRAMES / video_duration
        target_fps = min(target_fps, original_fps)

        print(f"Video {video_id}: Length {video_duration:.2f} sec. Set fps = {target_fps:.3f}")

        transcript_path = INPUT_PATH + "sf20k_{}_test_audios_vocals/".format(TYPE) + video_id + '/vocals_mono_text_segment.txt'
        transcript_text = ""
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as transcript_file:
                transcript_text = transcript_file.read()

        print("Length of transcript text: {}".format(len(transcript_text)))
        print("Sample: {}".format(transcript_text[:100]))

        abs_video_path = os.path.abspath(f)

        for index, row in part.iterrows():
            q = row['question']
            qid = row['question_id']
            print("Q:", q)

            cache_file = out_cache_folder + video_id + '_{}_qid_{}_results_{}_{}_char.txt'.format(TYPE, qid, N_FRAMES, model_name)

            process = True
            if os.path.isfile(cache_file):
                print("Restore from cache!")
                model_prediction = open(cache_file, 'r', encoding='utf8').read()
                if "</think>" in model_prediction:
                    print("Thinking complete!")
                    process = False

            if process:
                prompt = f"""
                    Based on the video and its transcript, give short and concise answer for the question.
                    The timestamps in the text correspond to the time in the video.
                    Answer must be less than 18 words.
    
                    Video transcript:
                    {transcript_text}
    
                    Question: {q}
                """

                # prompt = q

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "video_url",
                                "video_url": {
                                    "url": f"file://{abs_video_path}",
                                    "fps": target_fps,
                                    "min_pixels": 4 * 32 * 32,
                                    "max_pixels": 360 * 420
                                }
                            }
                        ],
                    }
                ]
                # print(messages)

                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=1.0,
                        top_p=0.95,
                        presence_penalty=0.0,
                        max_tokens=8 * 4096,
                        extra_body={
                            "mm_processor_kwargs": {
                                "fps": target_fps,
                                "min_pixels": 4 * 32 * 32,
                                "max_pixels": 360 * 420
                            },
                            "chat_template_kwargs": {
                                "enable_thinking": True,  # on by default
                                "preserve_thinking": True,  # on by default
                            },
                        },
                        reasoning_effort="xhigh",
                    )

                    model_prediction = response.choices[0].message.content.strip()

                except Exception as e:
                    print(f"Error for question {qid} for video {video_id}: {e}")
                    model_prediction = ""

                model_prediction = model_prediction.strip()
                out = open(cache_file, 'w', encoding='utf-8')
                out.write(model_prediction)
                out.close()

            if "</think>" in model_prediction:
                output_text = model_prediction.split("</think>")[-1].strip()
            else:
                output_text = model_prediction

            results.append({
                'question_id': qid,
                'prediction': output_text,
            })
            print("A:", output_text)

    with open(out_file, 'w', encoding='utf8') as out:
        out.write(json.dumps(results, ensure_ascii=False))
    print("Results were written to: ", out_file)


if __name__ == "__main__":
    print("Input folder:", INPUT_PATH)
    proc_with_qwen()