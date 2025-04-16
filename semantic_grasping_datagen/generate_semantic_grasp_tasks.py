import os.path
from collections import defaultdict
import json

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


initial_tasks_with_grasp_info = [
    "Pour coffee from the mug (handle grasp)",
    "Warm yourself with a hot mug (body grasp)",
    "Check the brand on the bottom of the mug (rim grasp to flip)",
    "Drink a hot beverage comfortably (side grasp)",
    "Slice bread for sandwiches (knife handle grasp)",
    "Hand a knife safely to another person (blade hold grasp)",
    "Stabilize knife for precision cutting (knife tip and handle grasp)",
    "Set the fork properly on the table (fork tine manipulation)",
    "Eat (fork or spoon handle grip)",
    "Serve soup from the pot (pot handle grasp)",
    "Peel a banana (stem pinch grasp)",
    "Open a shopping bag (bag opening grasp)",
    "Carry the bag (handle loop grasp)",
    "Open a beer bottle (bottle neck or handle grasp)",
    "Pour beer into a glass (bottle body grasp)",
    "Read the book (book edge pinch)",
    "Re-shelve the book (book spine grasp)",
    "Unscrew a bottle cap (cap twist grasp)",
    "Pour liquid from a bottle (bottle neck grasp)",
    "Serve cereal from a bowl (bowl rim grasp)",
    "Mix ingredients (any food item plus a bowl)",
    "Position camera for a photo (DSLR body grasp)",
    "Adjust camera zoom lens (lens barrel rotation grasp)",
    "Drip wax onto table (candle side grasp)",
    "Open a canister lid (lid twist grasp)",
    "Rotate can opener around can (handle rotation grasp)",
    "Adjust cap fit on head (brim grasp and rotation)",
    "Take a selfie (side edge grasp - cell phone)",
    "Pour cereal from box (box side panel grasp)",
    "Open cereal box (flap pinch and pull)",
    "Break chocolate into pieces (chocolate edge grasp)",
    "Stack coasters (coaster edge pinch)",
    "Move computer mouse (mouse body grasp)",
    "Hold gaming controller (controller side grips)",
    "Sip water (cup body grasp)",
    "Adjust desk lamp position (lamp neck grasp)",
    "Squeeze drink bottle (bottle body compression)",
    "Open sports bottle cap (cap flip grasp)",
    "Stir drink (utensil handle grasp)",
    "Change DSLR camera lens (lens twist)",
    "Point flashlight at target (flashlight body grasp)",
    "Put on glasses (glasses stem grasp)",
    "Clean glasses lenses (glasses frame edge grasp)",
    "Tune the guitar (tuning peg rotation grasp)",
    "Strum the guitar (pick grasp or finger positioning)",
    "Use hammer with maximum leverage (hammer handle grasp - near base)",
    "Tap very gently on the table with the hammer  (hammer handle grasp - near head)",
    "Hang clothing on a hanger (hanger hook grasp)",
    "Put on hat (brim manipulation)",
    "Hang up headphones (headband grasp)",
    "Cut paper (scissors handle grasp)",
    "Spread jam with a knife (knife flat blade grasp)",
    "Open laptop lid (lid edge grasp)",
    "Write with marker on whiteboard (marker body grasp)",
    "Pour milk (carton side or handle grasp)",
    "Open milk (spout manipulation)",
    "Flip notepad to clean page (notepad edge grasp)",
    "Toss pancakes in a pan (pan handle grasp with wrist motion)",
    "Open box (box flap grasp)",
    "Bind pages with paperclip (paperclip pinch and slide)",
    "Write signature (pen body grasp)",
    "Open pill bottle (cap press and turn)",
    "Bring plate to the dining table (plate rim grasp)",
    "Open purse (purse clasp manipulation)",
    "Tune radio (knob rotation grasp)",
    "Put ring on finger (ring exterior grasp)",
    "Polish ring (ring secure grasp)",
    "Measure length with ruler (ruler alignment grasp)",
    "Cut fabric with scissors (scissors handle grasp)",
    "Tighten screw with screwdriver (screwdriver handle grip with rotation)",
    "Squeeze shampoo onto palm (bottle compression grasp)",
    "Tie shoelaces (lace grasp)",
    "Lather soap (soap surface grasp)",
    "Pump soap (pump press with palm)",
    "Open pull-tab on soda can (tab pinch and lift)",
    "Pour soda into glass (can angle control grasp)",
    "Stir hot drink with spoon (spoon handle grasp)",
    "Silence alarm (button manipulation)",
    "Put the laptop away (lid center grasp)",
    "Pour tea from teapot (teapot handle grasp)",
    "Check if the teapot needs a refill (lid knob grasp)",
    "Insert USB stick into port (USB body grasp)",
    "Toast with wine glass (stem secure grasp)",
]


def _make_category_to_tasks():
    mapping_rules = {
        "banana": "Banana",
        "bag": "Bag",
        "beer": "BeerBottle",
        "book": "Book",
        "bottle": "Bottle",
        "bowl": "Bowl",
        "bread": "BreadSlice",
        "camera": "DSLRCamera",
        "candle": "Candle",
        "canister": "Canister",
        "can opener": "CanOpener",
        "cap": "Cap",
        "cell phone": "CellPhone",
        "cereal box": "CerealBox",
        "chocolate": "Chocolate",
        "coaster": "Coaster",
        "mouse": "ComputerMouse",
        "controller": "Controller",
        "cup": "Cup",
        "desk lamp": "DeskLamp",
        "drink bottle": "DrinkBottle",
        "dslr": "DSLRCamera",
        "flashlight": "Flashlight",
        "fork": "Fork",
        "glasses": "Glasses",
        "guitar": "Guitar",
        "hammer": "Hammer",
        "hanger": "Hanger",
        "hat": "Hat",
        "headphones": "Headphones",
        "knife": "Knife",
        "laptop": "Laptop",
        "marker": "Marker",
        "milk": "MilkCarton",
        "mug": "Mug",
        "notepad": "Notepad",
        "pan": "Pan",
        "box": "PaperBox",
        "paperclip": "PaperClip",
        "pen": "Pen",
        "pill": "PillBottle",
        "plate": "Plate",
        "purse": "Purse",
        "radio": "Radio",
        "ring": "Ring",
        "ruler": "Ruler",
        "scissors": "Scissors",
        "screwdriver": "ScrewDriver",
        "shampoo": "Shampoo",
        "soap": "SoapBottle",
        "soda can": "SodaCan",
        "spoon": "Spoon",
        "teapot": "Teapot",
        "usb": "USBStick",
        "wine glass": "WineGlass",
    }

    # Create mapping from object to tasks
    object_to_tasks = defaultdict(list)
    for task in initial_tasks_with_grasp_info:
        matched = False
        lower_task = task.lower()
        for keyword, obj_type in mapping_rules.items():
            if keyword.lower() in lower_task:
                object_to_tasks[obj_type].append(task)
                matched = True
        if not matched:
            object_to_tasks["Unknown"].append(task)

    return dict(object_to_tasks)


category_to_tasks = {
    "Mug": [
        "Pour coffee from the mug (handle grasp)",
        "Warm yourself with a hot mug (body grasp)",
        "Check the brand on the bottom of the mug (rim grasp to flip)",
        "Drink a hot beverage comfortably (side grasp)",
    ],
    "Unknown": [
        "Serve soup from the pot (pot handle grasp)",
        "Stir drink (utensil handle grasp)",
        "Tie shoelaces (lace grasp)",
        "Silence alarm (button manipulation)",
    ],
    "Knife": [
        "Slice bread for sandwiches (knife handle grasp)",
        "Hand a knife safely to another person (blade hold grasp)",
        "Stabilize knife for precision cutting (knife tip and handle grasp)",
        "Spread jam with a knife (knife flat blade grasp)",
    ],
    "Fork": [
        "Set the fork properly on the table (fork tine manipulation)",
        "Eat (fork or spoon handle grip)",
    ],
    "Spoon": [
        "Eat (fork or spoon handle grip)",
        "Stir hot drink with spoon (spoon handle grasp)",
    ],
    "Banana": [
        "Peel a banana (stem pinch grasp)",
    ],
    "Bag": [
        "Open a shopping bag (bag opening grasp)",
        "Carry the bag (handle loop grasp)",
    ],
    "Pen": [
        "Write signature (pen body grasp)",
    ],
    "BeerBottle": [
        "Open a beer bottle (bottle neck or handle grasp)",
        "Pour beer into a glass (bottle body grasp)",
    ],
    "Bottle": [
        "Unscrew a bottle cap (cap twist grasp)",
        "Pour liquid from a bottle (bottle neck grasp)",
        "Squeeze drink bottle (bottle body compression)",
        "Open sports bottle cap (cap flip grasp)",
        "Open pill bottle (cap press and turn)",
        "Squeeze shampoo onto palm (bottle compression grasp)",
    ],
    "Book": [
        "Read the book (book edge pinch)",
        "Re-shelve the book (book spine grasp)",
    ],
    "Cap": [
        "Unscrew a bottle cap (cap twist grasp)",
        "Adjust cap fit on head (brim grasp and rotation)",
        "Open sports bottle cap (cap flip grasp)",
    ],
    "Bowl": [
        "Serve cereal from a bowl (bowl rim grasp)",
        "Mix ingredients (any food item plus a bowl)",
    ],
    "DSLRCamera": [
        "Position camera for a photo (DSLR body grasp)",
        "Position camera for a photo (DSLR body grasp)",
        "Adjust camera zoom lens (lens barrel rotation grasp)",
        "Change DSLR camera lens (lens twist)",
        "Change DSLR camera lens (lens twist)",
    ],
    "Candle": [
        "Drip wax onto table (candle side grasp)",
    ],
    "Canister": [
        "Open a canister lid (lid twist grasp)",
    ],
    "CanOpener": [
        "Rotate can opener around can (handle rotation grasp)",
    ],
    "CellPhone": [
        "Take a selfie (side edge grasp - cell phone)",
    ],
    "Pan": [
        "Toss pancakes in a pan (pan handle grasp with wrist motion)",
    ],
    "PaperBox": [
        "Open box (box flap grasp)",
    ],
    "CerealBox": [
        "Pour cereal from box (box side panel grasp)",
        "Open cereal box (flap pinch and pull)",
    ],
    "Chocolate": [
        "Break chocolate into pieces (chocolate edge grasp)",
    ],
    "Coaster": [
        "Stack coasters (coaster edge pinch)",
    ],
    "ComputerMouse": [
        "Move computer mouse (mouse body grasp)",
    ],
    "Controller": [
        "Hold gaming controller (controller side grips)",
    ],
    "Cup": [
        "Sip water (cup body grasp)",
    ],
    "DeskLamp": [
        "Adjust desk lamp position (lamp neck grasp)",
    ],
    "DrinkBottle": [
        "Squeeze drink bottle (bottle body compression)",
    ],
    "Flashlight": [
        "Point flashlight at target (flashlight body grasp)",
    ],
    "Glasses": [
        "Put on glasses (glasses stem grasp)",
        "Clean glasses lenses (glasses frame edge grasp)",
    ],
    "Guitar": [
        "Tune the guitar (tuning peg rotation grasp)",
        "Strum the guitar (pick grasp or finger positioning)",
    ],
    "Hammer": [
        "Use hammer with maximum leverage (hammer handle grasp - near base)",
        "Tap very gently on the table with the hammer  (hammer handle grasp - near head)",
    ],
    "Hanger": [
        "Hang clothing on a hanger (hanger hook grasp)",
    ],
    "Hat": [
        "Put on hat (brim manipulation)",
    ],
    "Headphones": [
        "Hang up headphones (headband grasp)",
    ],
    "Scissors": [
        "Cut paper (scissors handle grasp)",
        "Cut fabric with scissors (scissors handle grasp)",
    ],
    "Laptop": [
        "Open laptop lid (lid edge grasp)",
        "Put the laptop away (lid center grasp)",
    ],
    "Marker": [
        "Write with marker on whiteboard (marker body grasp)",
    ],
    "MilkCarton": [
        "Pour milk (carton side or handle grasp)",
        "Open milk (spout manipulation)",
    ],
    "Notepad": [
        "Flip notepad to clean page (notepad edge grasp)",
    ],
    "PaperClip": [
        "Bind pages with paperclip (paperclip pinch and slide)",
    ],
    "PillBottle": [
        "Open pill bottle (cap press and turn)",
    ],
    "Plate": [
        "Bring plate to the dining table (plate rim grasp)",
    ],
    "Ring": [
        "Bring plate to the dining table (plate rim grasp)",
        "Put ring on finger (ring exterior grasp)",
        "Polish ring (ring secure grasp)",
    ],
    "Purse": [
        "Open purse (purse clasp manipulation)",
    ],
    "Radio": [
        "Tune radio (knob rotation grasp)",
    ],
    "Ruler": [
        "Measure length with ruler (ruler alignment grasp)",
    ],
    "ScrewDriver": [
        "Tighten screw with screwdriver (screwdriver handle grip with rotation)"
    ],
    "Shampoo": [
        "Squeeze shampoo onto palm (bottle compression grasp)",
    ],
    "SoapBottle": [
        "Lather soap (soap surface grasp)",
        "Pump soap (pump press with palm)",
    ],
    "SodaCan": [
        "Open pull-tab on soda can (tab pinch and lift)",
        "Pour soda into glass (can angle control grasp)",
    ],
    "Teapot": [
        "Pour tea from teapot (teapot handle grasp)",
        "Check if the teapot needs a refill (lid knob grasp)",
    ],
    "USBStick": [
        "Insert USB stick into port (USB body grasp)",
    ],
    "WineGlass": [
        "Toast with wine glass (stem secure grasp)",
    ],
}


def get_missing_object_types(output_file, batch_size=5, implicit=True):
    unused_categories = set(all_categories)  # - set(category_to_tasks.keys())

    output_file = os.path.expanduser(output_file)

    if os.path.isfile(output_file):
        with open(output_file) as f:
            all_grasps = json.load(f)
    else:
        all_grasps = {}

    missing_categories = sorted(list(unused_categories - set(all_grasps.keys())))

    if len(missing_categories) == 0:
        return all_grasps

    llm = LangchainWrapper(
        ChatOpenAI(
            model=MODEL_NAME,
            max_tokens=4096,
        )
    )

    if implicit:
        implicit_str = (
            " without explicitly providing instructions about the grasp description"
            " (like object part to contact or direction to approach)"
        )
    else:
        implicit_str = ""

    for first_category in range(0, len(missing_categories), batch_size):
        cur_categories = missing_categories[
            first_category : first_category + batch_size
        ]
        print(first_category, len(missing_categories), cur_categories)

        query = (
            "For each object type in the list below, we need to make a short list of up to 5 grasp definitions"
            " (part of the object to grasp with a single 6-DOF end effector, and relative orientation of"
            " the gripper with respect to the part to grasp). Try to make the grasps as varied as possible."
            " Assuming that the objects from each category are lying on a table or other surface, avoid grasps"
            " that assume that the object needs to be approached from underneath or placed upright while holding"
            " it from underneath. Do not assume the presence of any optional feature in an object of the given"
            " category. If some object is too generic to identify grasps (e.g. an undetermined food item, or an"
            " unidentified piece of fruit), feel free to return an empty list."
            " Then, for each of the identified grasps, generate 4 semantic grasping tasks that require that"
            " type of grasp (and not any other from the list) to correctly hold the object towards task"
            f" completion{implicit_str}. The task definition might require a second gripper for completion,"
            " but the grasp should be possible with a single gripper. Generate the output as a JSON map"
            " from object type to a list of dicts, each with a"
            " `grasp_definition` (natural language str) and a `semantic_tasks` short list of 4 task descriptions"
            " (each a natural language str). Do not add any additional comments. The list of object types"
            f" is\n\n{cur_categories}"
        )

        ans = llm(query, log="grasps")

        grasps = llm.extract_json(ans)
        if grasps is None:
            continue

        all_grasps.update(grasps)

        with open(output_file, "w") as f:
            json.dump(all_grasps, f, indent=2)

    llm.print_costs()

    return all_grasps


def get_sparser_grasps(all_grasps, output_file):
    output_file = os.path.expanduser(output_file)

    if os.path.isfile(output_file):
        with open(output_file) as f:
            sparse_grasps = json.load(f)
    else:
        sparse_grasps = {}

    remaining_categories = sorted(
        list(set(all_grasps.keys()) - set(sparse_grasps.keys()))
    )

    if len(remaining_categories) == 0:
        return sparse_grasps

    llm = LangchainWrapper(
        ChatOpenAI(
            model=MODEL_NAME,
            max_tokens=4096,
        )
    )

    for it, category in enumerate(remaining_categories):
        original_grasps = all_grasps.get(category, [])
        if not original_grasps:
            print(f"No grasps defined for {category}")
            sparse_grasps[category] = []
            continue

        print(it, len(remaining_categories), category)

        grasp_list_str = "\n".join(
            [f"- {g['grasp_definition']}" for g in original_grasps]
        )

        query = (
            f"Given the grips of a {category} defined by:\n\n"
            f"{grasp_list_str}\n\n"
            "Are they all significantly different among them? If some grasp seems not generally applicable"
            " to an object of the given category as it perhaps assumes the presence of an optional feature in the"
            " object, we should discard that type of grasp. Assuming that the objects from each category are lying"
            " on a table or other surface, avoid grasps that assume that the object needs to be approached from"
            " underneath or placed upright while holding it from underneath."
            " Feel free to briefly discuss and reason, and then"
            " list the up to three generally applicable grasps that have least resemblance among them"
            " as a JSON parsable list of strings."
        )

        ans = llm(query, log="sparser_grasps")

        sparse_set = llm.extract_json(ans)
        if sparse_set is None:
            continue

        sparse_grasps[category] = sparse_set

        with open(output_file, "w") as f:
            json.dump(sparse_grasps, f, indent=2)

    with open(output_file, "w") as f:
        json.dump(sparse_grasps, f, indent=2)

    llm.print_costs()

    return sparse_grasps


def filter_with_semantic_task_coverage(all_grasps, sparse_grasps, output_file):
    output_file = os.path.expanduser(output_file)

    if os.path.isfile(output_file):
        with open(output_file) as f:
            coverage_results = json.load(f)
    else:
        coverage_results = {}

    remaining_categories = sorted(
        list(set(sparse_grasps.keys()) - set(coverage_results.keys()))
    )

    if len(remaining_categories) == 0:
        return coverage_results

    llm = LangchainWrapper(
        ChatOpenAI(
            model=MODEL_NAME,
            max_tokens=4096,
        )
    )

    for it, category in enumerate(remaining_categories):
        print(it, len(remaining_categories), category)

        if not all_grasps[category]:
            coverage_results[category] = {}
            print(f"No grasps for {category}")
            continue

        sparse_set = sparse_grasps.get(category, [])
        all_grasps_in_cat = all_grasps.get(category, [])

        if len(sparse_set) < 2:
            coverage_results[category] = {}
            print(f"Only {len(sparse_set)} grasps for {category}")
            continue

        semantic_tasks = []
        full_set = []
        for g in all_grasps_in_cat:
            full_set.append(g["grasp_definition"])
            semantic_tasks.extend(g.get("semantic_tasks", []))

        semantic_tasks_text = "\n".join(f" - {t}" for t in semantic_tasks)

        index_to_grasp = {it: grasp for it, grasp in enumerate(sparse_set)}
        # index_to_grasp = {it: grasp for it, grasp in enumerate(full_set)}
        grasp_str = "\n".join(
            f" {it} {index_to_grasp[it]}" for it in range(len(index_to_grasp))
        )

        query = (
            f"For an object of category '{category}' with the following possible grasps:\n\n"
            f"{grasp_str}\n\n"
            "that are identified by their numerical indices,"
            f" which of the following semantic tasks seem solvable by each grasp?\n\n"
            f"{semantic_tasks_text}\n\n"
            "Discard semantic tasks that assume the presence of features not generally found in objects of this"
            " category. You may briefly discuss any ambiguities or decisions, and finally add a JSON parsable"
            " dict from each kept semantic task to the corresponding list of valid grasp indices."
        )

        ans = llm(query, log="semantic_task_coverage")
        coverage_dict = llm.extract_json(ans)

        if coverage_dict is None:
            continue

        current_valid_grasps = {}

        num_valid_tasks = 0
        num_valid_in_discarded_grasps = 0

        for semantic_task, valid_grasps in coverage_dict.items():
            if len(valid_grasps) > 1:
                continue

            grasp = index_to_grasp[valid_grasps[0]]
            if grasp not in sparse_set:
                num_valid_in_discarded_grasps += 1

            num_valid_tasks += 1

            if grasp in current_valid_grasps:
                current_valid_grasps[grasp].append(semantic_task)
            else:
                current_valid_grasps[grasp] = [semantic_task]

        coverage_results[category] = current_valid_grasps

        print(
            category,
            num_valid_tasks,
            "accepted",
            num_valid_in_discarded_grasps,
            "in discarded grasps",
        )

        with open(output_file, "w") as f:
            json.dump(coverage_results, f, indent=2)

    with open(output_file, "w") as f:
        json.dump(coverage_results, f, indent=2)

    llm.print_costs()
    return coverage_results


if __name__ == "__main__":

    def main():
        implicit = True

        implicit_str = "_implicit" if implicit else ""

        all_grasps = get_missing_object_types(
            f"~/Desktop/semantic_grasps{implicit_str}.json", implicit=implicit
        )

        sparse_grasps = get_sparser_grasps(
            all_grasps, output_file=f"~/Desktop/sparse_grasps{implicit_str}.json"
        )
        coverage = filter_with_semantic_task_coverage(
            all_grasps,
            sparse_grasps,
            output_file=f"~/Desktop/semantic_task_after_coverage{implicit_str}.json",
        )

    main()
    print("DONE")
