import argparse
import json
import os
from threading import Semaphore
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from pydantic import BaseModel

SYS_PROMPT = """
You are a helpful assistant that helps reword task descriptions to be more natural.

You will be given a simple task description, and you will need to reword it to be more natural. Level 1 and level 2 tasks should be natural sentences that a human would say.
The simple task description will be Level 0, and you should provide a Level 1 and Level 2 rewording.
Level 1 should be a little less object-centric, and should focus on the action.
Level 2 should be even less object-centric, and should be an abstract command a human would give to a robot. You don't have to say the name of the object if it's obvious from the context. You may ground in another object if needed to make it more natural.

Here's an example:
Level 0: Grasp the spoon to stir
Level 1: Stir a bowl of soup
Level 2: Mix the soup

Prompted with the level 0 task, you should respond in JSON format with the following keys:
- level_1: a more natural rewording of the level 0 task
- level_2: a more natural rewording of the level 1 task
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks_file")
    parser.add_argument("out_dir")
    parser.add_argument("--n-prefetch", type=int, default=1)
    return parser.parse_args()

class TaskRewording(BaseModel):
    level_1: str
    level_2: str

def reword(client: OpenAI, semaphore: Semaphore, noun: str, verb: str):
    semaphore.acquire()
    level_0 = f"Grasp the {noun} to {verb}"
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": f"The level 0 task is: {level_0}"}
        ],
        response_format=TaskRewording
    )
    rewording: TaskRewording = response.choices[0].message.parsed
    data = {
        "level_0": level_0,
        "level_1": rewording.level_1,
        "level_2": rewording.level_2
    }
    return noun, verb, data

def main():
    args = get_args()
    client = OpenAI()

    os.makedirs(args.out_dir, exist_ok=True)
    succ_file = os.path.join(args.out_dir, "reworded_tasks.json")
    fail_file = os.path.join(args.out_dir, "failed_tasks.json")

    with open(args.tasks_file, "r") as f:
        lines = f.read().strip().splitlines()

    obj_tasks = set()
    for line in lines:
        obj, verb, label = line.split("-")
        if label == "True":
            obj = obj.split("_", 1)[1]
            obj = obj.replace("_", " ")
            obj_tasks.add((obj, verb))
    obj_tasks = list(obj_tasks)

    if os.path.exists(succ_file):
        with open(succ_file, "r") as f:
            succ_tasks: dict[str, dict[str, str]] = json.load(f)
        completed = set(tuple(k.split("-")) for k in succ_tasks.keys())
        obj_tasks = [t for t in obj_tasks if t not in completed]
    else:
        succ_tasks = {}

    if os.path.exists(fail_file):
        with open(fail_file, "r") as f:
            failed_tasks: dict[str, dict[str, str]] = json.load(f)
        failed = set(tuple(k.split("-")) for k in failed_tasks.keys())
        obj_tasks = [t for t in obj_tasks if t not in failed]
    else:
        failed_tasks = {}

    print(f"Generating rewordings for {len(obj_tasks)} object-task pairs...")
    fetch_semaphore = Semaphore(args.n_prefetch)
    with ThreadPoolExecutor(max_workers=args.n_prefetch) as executor:
        futures = [executor.submit(reword, client, fetch_semaphore, noun, verb) for noun, verb in obj_tasks]
        for i, future in enumerate(as_completed(futures)):
            fetch_semaphore.release()
            noun, verb, data = future.result()
            print(f"{i+1}/{len(obj_tasks)}: {noun}, {verb}")
            print(f"\tLevel 0: {data['level_0']}")
            print(f"\tLevel 1: {data['level_1']}")
            print(f"\tLevel 2: {data['level_2']}")
            while (approve := input("Approve? (y/n): ")) not in ["y", "n"]:
                pass
            if approve == "y":
                succ_tasks[f"{noun}-{verb}"] = data
                with open(succ_file, "w") as f:
                    json.dump(succ_tasks, f, indent=2)
            else:
                failed_tasks[f"{noun}-{verb}"] = data
                with open(fail_file, "w") as f:
                    json.dump(failed_tasks, f, indent=2)

if __name__ == "__main__":
    main()
