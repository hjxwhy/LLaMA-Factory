import json
import os
from openai import OpenAI
from tqdm import tqdm
api_key = "None"

system_prompt = """你是一个数据生成专家，你擅长生成一些数据。我会给你一句话，这句话表示机械臂能完成的一个动作，我需要你生成三个新的错误选项跟这个任务相近的。你可以从一下方面扩充，不限制这五个方面，你可以自由发挥：
1. 机械臂动作替换，动词替换
2.抓取目标替换
3.放置目标位置
4.顺序替换
5.截取动作的一半，没有完成任务

随机选择几项创造出三个错误的选项，以json的格式返回。
返回格式：
[{"option1": <fake option>} ...]"""

client = OpenAI(api_key=api_key)

with open("/localfolder/code/LLaMA-Factory/track_infer_dataset.json", "r") as f:
    track_infer_dataset = json.load(f)

for example in tqdm(track_infer_dataset):
    try:
        task = example["messages"][-1]["content"]
        task = task.split("The task is: ")[1]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ]
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
        )
        content = response.choices[0].message.content
        example["fake_task"] = content
    except Exception as e:
        print(e)
        continue

with open("/localfolder/code/LLaMA-Factory/track_infer_dataset_fake_task.json", "w") as f:
    json.dump(track_infer_dataset, f, indent=4)
