import copy
import os.path
import json
from collections import defaultdict

import numpy as np

from semantic_grasping_datagen.langchain_wrapper import LangchainWrapper, ChatOpenAI

MODEL_NAME = "gpt-4.1"

all_categories = [
    "Banana",
    "Bag",
    "BeerBottle",
    "Book",
    "Bottle",
    "Bowl",
    "BreadSlice",
    "Calculator",
    "Camera",
    "Candle",
    "Canister",
    "CanOpener",
    "Cap",
    "Carrot",
    "Cassette",
    "CellPhone",
    "CerealBox",
    "Chocolate",
    "Coaster",
    "Coin",
    "ComputerMouse",
    "Controller",
    "Cookie",
    "Cup",
    "CupCake",
    "DeskLamp",
    "DiscCase",
    "Donut",
    "DrinkBottle",
    "DrinkingUtensil",
    "DSLRCamera",
    "Eraser",
    "Flashlight",
    "FoodItem",
    "Fork",
    "Fruit",
    "Glasses",
    "Guitar",
    "Hammer",
    "Hanger",
    "Hat",
    "Headphones",
    "Keyboard",
    "Knife",
    "Laptop",
    "Magnet",
    "Marker",
    "MediaDiscs",
    "MilkCarton",
    "MousePad",
    "Mug",
    "NintendoDS",
    "Notepad",
    "Pan",
    "PaperBox",
    "PaperClip",
    "Pen",
    "Pencil",
    "PictureFrame",
    "PillBottle",
    "Plate",
    "PowerStrip",
    "Purse",
    "Radio",
    "Ring",
    "RubiksCube",
    "Ruler",
    "Scissors",
    "ScrewDriver",
    "Shampoo",
    "Shoes",
    "SoapBar",
    "SoapBottle",
    "SodaCan",
    "Spoon",
    "Stapler",
    "TableClock",
    "TableLamp",
    "TapeMeasure",
    "Teacup",
    "Teapot",
    "ToiletPaper",
    "USBStick",
    "VideoGameController",
    "WallClock",
    "Wallet",
    "Watch",
    "WebCam",
    "Wii",
    "WineBottle",
    "WineGlass",
]


def generate_grasps(output_file):
    output_file = os.path.expanduser(output_file)

    if os.path.isfile(output_file):
        with open(output_file) as f:
            all_grasps = json.load(f)
    else:
        all_grasps = {}

    missing_categories = sorted(list(set(all_categories) - set(all_grasps.keys())))

    if len(missing_categories) == 0:
        return all_grasps

    llm = LangchainWrapper(
        ChatOpenAI(
            model=MODEL_NAME,
            max_tokens=4096,
        )
    )

    how_many = "two"

    example = {
        "object_subtypes": [
            "Cordless drill",
            "Pneumatic drill",
            "Hammer drill",
            "Handheld electric drill",
        ],
        "optional_parts": ["Auxiliary handle", "Battery pack", "Belt clip", "Power cable"],
        "common_parts": ["Handle", "Motor body", "Chuck", "Trigger"],
        "object_table_contact_parts": ["Handle", "Motor body"],
        "common_graspable_parts": ["Handle", "Motor body"],
        "grasps": [
            {
                "object_part": "Handle",
                "example_task": "operating the drill to make holes into a wall",
                "approach_direction": "from the side",
                "finger_plane": "up/down relative to the arm's axis",
                "gripper_orientation": "sideways relative to the surface",
                "natural_language": "Grasp around the handle from the side with the gripper oriented sideways to align with the slim, elongated shape of the handle.",
            },
            {
                "object_part": "Motor body",
                "example_task": "lifting and moving the drill to clean the table surface",
                "approach_direction": "from above",
                "finger_plane": "left/right relative to the arm's axis",
                "gripper_orientation": "downward-facing toward the surface",
                "natural_language": "Grasp around the motor body from above with the gripper facing downward to enclose the broader, bulkier section securely.",
            },
        ],
    }

    for itc, category in enumerate(missing_categories):
        print(itc, len(missing_categories), category)

        query = (
            f"For objects of a given category, we need to determine {how_many} grasp definitions"
            " considering a single 6-DOF end effector that are as varied as possible (e.g., for"
            " ladles we would probably define one grasp around the handle and another one around the bowl,"
            " possibly requiring different relative orientations of the gripper with respect to the object part) and are"
            " reasonable for specific tasks/contexts (think about different grasps when using, cleaning, or handing a knife)."
            " Assuming that the objects from each category are standing or lying on a table or similar surface, avoid grasps"
            " that assume that the object needs to be approached from underneath or example tasks that require the"
            " object to be placed upright while holding it from underneath for the target object pose on some"
            " surface. Do not assume"
            " the presence of any optional feature in an object of the given category (e.g. some chair subtypes might"
            " have legs, but others a wheeled base instead, while all chairs have a back rest and a seat)."
            " Related, if the object category is too generic to identify object parts (e.g. an undetermined `tool`),"
            " feel free to return an empty list. Generate the output as a JSON dict with:\n"
            " - `object_subtypes` (list of str, possibly empty) different types of object for the given category (which might include specific optional parts),\n"
            " - `optional_parts` (list of str, possibly empty) present in only some objects of the category and subtypes and should be avoided\n"
            " - `common_parts` (list of str, possibly empty) reasonably comprehensive list of parts of any object of the given category (no optional ones). Here you can merge parts that might be differently names in subclasses but offer a common affordance for any object of the category or subtype(s)\n"
            " - `object_table_contact_parts` (list of str) part(s) of object that are assumed to be in contact with the table or surface underneath in the default starting pose (this defines a starting object orientation). If more than one part, make sure these are plausibly simultaneously in contact with the underlying surface,\n"
            " - `common_graspable_parts` (list of str, possibly empty) that can be used to generate grasps for any object of the category\n"
            f" - `grasps`, a list of {how_many} dicts with the entries:\n"
            f"   - an `object_part` (str),\n"
            f"   - an `example_task`(str),\n"
            f"   - an `approach_direction` (str, for wrist axis, e.g. from above, from the side, from below, at an angle, if relevant, else `Any`, relative to the orientation implied by the object-table contact parts),\n"
            f"   - a `finger_plane` (str, relative to wrist axis, e.g. left/right or up/down relative to the arm's axis, if relevant, else `Any`),\n"
            f"   - a `gripper_orientation` (str, whether the gripper faces up, down, sideways, or is diagonal/angled tilted, if relevant, else `Any`),\n"
            f"   - and a `natural_language` (str, description of the grasp in natural language, avoiding any reference to the example task and avoiding irrelevant grasp parameters, if any)\n"
            f"\nMake sure that example tasks do not state the object part to contact or the direction to approach,"
            f" and are unfeasible for the alternative grasp(s) in the list. If these requirements seem impossible to fulfill,"
            f" it is best to return an empty list of grasps. Be very descriptive about the relative gripper orientations.\n"
            f" One example for the `drill` category would be:\n\n{json.dumps(example, indent=2)}\n"
            f"Feel free to discuss options to annotate the object type while fulfilling the requirements before"
            f" generating the JSON dict. The object type to annotate is {category}."
        )

        ans = llm(query, log="grasps")

        grasp_info = llm.extract_json(ans)
        if grasp_info is None:
            print(f"Failed to parse LLM response for category '{category}'")
            continue

        all_grasps[category] = grasp_info

        print(json.dumps(grasp_info["grasps"], indent=2))

        with open(output_file, "w") as f:
            json.dump(all_grasps, f, indent=2)

    llm.print_costs()

    return all_grasps


def semantic_tasks_from_grasps(all_grasp_infos, output_file):
    output_file = os.path.expanduser(output_file)

    if os.path.isfile(output_file):
        with open(output_file) as f:
            semantic_tasks = json.load(f)
    else:
        semantic_tasks = {}

    remaining_categories = sorted(
        list(set(all_categories) - set(semantic_tasks.keys()))
    )

    if len(remaining_categories) == 0:
        return semantic_tasks

    llm = LangchainWrapper(
        ChatOpenAI(
            model=MODEL_NAME,
            max_tokens=4096,
        )
    )

    for it, category in enumerate(remaining_categories):
        grasp_infos = all_grasp_infos[category]["grasps"]
        object_contacts = all_grasp_infos[category]["object_table_contact_parts"]
        if not grasp_infos:
            print(f"No grasp data for {category}")
            semantic_tasks[category] = {}
            continue

        print(it, len(remaining_categories), category)

        prompt = (
            f" We need to generate semantic manipulation tasks requiring each of the given grasps in the list provided"
            f" at the end. Please generate the tasks for each grasp with the following design criteria, where each"
            f" criterion is first identified by a short name and then described in more detail:\n"
            "1. Clear target. Ensure that every task mentions the object type (e.g., 'the mug') unless it is obvious without it.\n"
            "2. Unknown state. Avoid tasks that make assumptions about the state of the object (e.g. being open/closed, empty/full, etc.).\n"
            "3. Unknown context. Avoid tasks that make assumptions about the surroundings/context of the object (i.e. assuming the presence of any other objects of the same category or others in the scene, other than the presence of a table top or similar surface underneath the object at the start).\n"
            "4. Implicit grasp. Avoid references to the part of the object being grasped (e.g., 'by the handle') or any of the grasp definition parameters in the task definition.\n"
            "5. Single gripper. While you should favor single-gripper task definitions, if a second gripper is implied or required, it should not be assumed to be present for the initial grasp, but rather during a subsequent step (e.g. if 'while another gripper does ...' seems reasonable, convert it into 'for posterior ...').\n"
            "6. Physical plausibility. Avoid tasks that require physically implausible configurations, like the object being placed standing on some surface while held from underneath.\n"
            "7. Compact instruction. Write tasks in compact and intelligible natural language and avoid technical formating like snake case.\n"
            "8. Semantic meaning. Avoid simple pick and place tasks, and try to focus on semantic tasks, i.e., they should rely on some affordance of the object or consider some compositional task where we must manipulate the object towards some meaningful goal.\n"
            "9. Identifiability. If both provided grasps, object category or parts to grasp seem too coarse/vague/hard to identify, avoid defining any task and favor an empty list of tasks for each grasp.\n"
            "Try to generate four valid semantic tasks per grasp, making sure that the tasks are incompatible with"
            " the alternative grasp for the object category (they should imply different use cases or affordances)."
            " For each generated semantic task we need a dict with the entries:\n"
            " - `text`: the semantic task instruction, without mentioning the grasped part or approach direction, and mentioning the target object if needed,\n"
            " - `num_grippers`: the number of grippers required to complete the semantic task,\n"
            " - `grasp_critique`: short string justifying the lack of validity of the assigned grasp towards completing the task,\n"
            " - `grasp_score`: validity score in range 0 (low) to 9 (high) based on the grasp_critique,\n"
            " - `alternative_grasp_critique`: short string justifying the possible validity of the alternative grasp towards completing the task,\n"
            " - `alternative_grasp_score`: validity score in range 0 to 9 according based on the alternative_grasp_critique,\n"
            " - `weakest_point`: short name (string) of the task design criterion point most poorly fulfilled,\n"
            " - `task_criteria_fulfilled`: score the fulfillment of the weakest point in the range 0 (poor) to 9 (perfect fulfillment)\n"
            "Feel free to reason about the problem and generate a JSON dictionary mapping each grasp"
            " id to the list of semantic task dicts."
        )

        prompt += (
            f"\n\nThe following are the valid grasp ids and corresponding info for an object of type '{category}'"
            f" assuming the object is in contact with the underlying surface through its part(s) {object_contacts}:\n"
        )

        id_to_grasp = {}
        for grasp_info in grasp_infos:
            to_keep = {
                "object_part",
                "approach_direction",
                "finger_plane",
                "gripper_orientation",
                "natural_language",
            }

            new_info = copy.deepcopy(grasp_info)
            for key in set(new_info.keys()) | to_keep:
                if key not in to_keep:
                    new_info.pop(key)
                else:
                    assert key in new_info

            object_part = new_info["object_part"].lower().strip()
            approach_direction = new_info["approach_direction"].lower().strip()
            id = f"{object_part} {approach_direction}"
            if id in id_to_grasp:
                id += " 2"

            id_to_grasp[id] = new_info

        prompt += f"{json.dumps(id_to_grasp, indent=2)}\n"

        response = llm(prompt, log="semantic_tasks_from_grasps")
        parsed = llm.extract_json(response)
        if parsed is None:
            print(f"Failed to parse LLM response for category '{category}'")
            continue

        to_save = {}
        for id in parsed:
            to_save[id] = {"info": id_to_grasp[id], "tasks": parsed[id]}

        print(json.dumps(parsed, indent=2))

        semantic_tasks[category] = to_save

        with open(output_file, "w") as f:
            json.dump(semantic_tasks, f, indent=2)

    llm.print_costs()

    return semantic_tasks


def print_stats(cleaned):
    num_cats = 0
    num_tasks = 0
    positive_scores = 0
    negative_scores = 0
    score_deltas = 0
    valid_categories = set()
    weakest_points = defaultdict(list)
    task_criteria_scores = []
    for category, grasp_to_info_tasks in cleaned.items():
        if len(grasp_to_info_tasks) == 0:
            continue
        num_cats += 1
        valid_categories.add(category)
        for grasp, info_tasks in grasp_to_info_tasks.items():
            tasks = info_tasks["tasks"]
            num_tasks += len(tasks)
            for task in tasks:
                positive_scores += task["grasp_score"]
                negative_scores += task["alternative_grasp_score"]
                score_deltas += task["grasp_score"] - task["alternative_grasp_score"]
                weakest_points[task["weakest_point"].lower()].append(int(task["task_criteria_fulfilled"]))
                task_criteria_scores.append(int(task["task_criteria_fulfilled"]))

    print(
        f"{num_cats} categories out of {len(cleaned)},"
        f" {num_tasks} tasks,"
        f" {num_tasks / num_cats:.1f} tasks/category"
    )
    print(f"Missing categories {set(cleaned.keys()) - valid_categories}")
    print(
        f"Mean positive score {positive_scores / num_tasks:.1f}"
        f", negative score {negative_scores / num_tasks:.1f}"
        f", delta score {score_deltas / num_tasks:.1f}"
    )

    print(f"Weakest defined task points (mean {np.mean(task_criteria_scores):.1f}):")
    weakest_keys = sorted(list(weakest_points.keys()), key=lambda x: len(weakest_points[x]), reverse=True)
    for weakest_key in weakest_keys:
        print(f"{len(weakest_points[weakest_key])} {weakest_key} {np.mean(weakest_points[weakest_key]):.1f}")


if __name__ == "__main__":

    def main():
        grasps = generate_grasps(f"~/Desktop/grasp_info.json")
        tasks = semantic_tasks_from_grasps(
            grasps, f"~/Desktop/semantic_tasks_from_grasps.json"
        )
        print_stats(tasks)

    main()
    print("DONE")
