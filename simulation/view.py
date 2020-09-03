import json
import pickle
from functools import partial
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

import model.evolution as evo
import model.pathfinding as path


def loadPickle(fname: str) -> Any:
    """Loads a pickle file.

    Args:
        fname (str): root name of the file to load

    Returns:
        Any: pickled content
    """

    with open("simulation/data/solutions/" + fname, "rb") as handle:
        return pickle.loads(handle.read())


def loadJson(fname: str) -> Dict[str, Any]:
    """Loads a pickle file.

    Args:
        fname (str): root name of the file to load

    Returns:
        Dict[str, Any]: json-ed content
    """

    with open("simulation/data/solutions/" + fname + ".json", "r") as f:
        return json.load(f)


def getCenterCircle(
    rect_pos: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]],
    step_half_size: int,
    circle_radius: int,
    coor: Tuple[int, int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Gets coordinates for drawing circle at the center of rectangle.

    Args:
        rect_pos (List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]):
            rectangle coordinate system
        step_half_size (int): half size of a rectangle step
        circle_radius (int): size of circle
        coor (Tuple[int, int]): 2D coordinate from array

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: box coordinates of circle
    """

    x, y = coor
    center = tuple((c + step_half_size for c in rect_pos[y][x][0]))
    c1 = tuple((start - circle_radius for start in center))
    c2 = tuple((end + circle_radius for end in center))

    return c1, c2


def createGif(fname: str, skip_rake: bool, climb: bool) -> None:
    """Creates gif animation that visualizes the solution.

    Args:
        fname (str): root name of the file to load
        skip_rake (bool): skips the raking part
        climb (bool): Climbing distance approach. If True, distance is measured
            with abs(current terrain number - next terrain number)
    """

    # TODO?: visualization only works on rake_solved maps
    # TODO?: we must specify climb bool value
    try:
        map_props = path.Map(fname)
        terrained_map = evo.loadMap(fname + "_ter")
        rake_solved = loadPickle(fname + "_rake")
        paths_solved = loadPickle(fname + "_path")
        rule_solved = loadJson(fname + "_rule")
    except FileNotFoundError as e:
        print(e)
        return

    # parameters
    step_size = 50  # should not be changed, other sizes are not scalled
    info_space = 350

    # get sizes of window, drawings, text, circles
    height = map_props.height * step_size
    width = map_props.width * step_size
    properties = map_props.properties
    map_width = width + info_space
    map_height = height + 1
    step_half_size = int(step_size / 2)
    circle_radius = int(step_size / 5)

    # get stepping color, color sat/lum
    all_hue_values = 180
    last_rake_value = tuple(rake_solved.values())[-1]
    color_step = int(all_hue_values / last_rake_value)
    saturation, luminance = 100, 50

    # create first image and font
    image = Image.new(mode="RGB", size=(map_width, map_height), color="white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial", step_half_size)
    small_font = ImageFont.truetype("arial", int(step_half_size / 2))
    small_font_bold = ImageFont.truetype("arialbd", int(step_half_size / 2))
    frames = [image]

    # draw window of empty rectangles and unpassable locations
    # also save rectangle coordinate system
    rect_pos = []  # type: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]
    for i, x in enumerate(range(0, width, step_size)):
        rect_pos.append([])
        for j, y in enumerate(range(0, height, step_size)):
            rect = (x, y), (x + step_size, y + step_size)
            rect_pos[i].append(rect)
            if terrained_map[j][i] == "-2":
                draw.rectangle(rect, fill=(0, 127, 255), outline="black")
            elif terrained_map[j][i] == "-1":
                draw.rectangle(rect, fill="black", outline="black")
            else:
                draw.rectangle(rect, outline="black")

    # draw text explanation
    row = 1
    text_h = row * 15
    text_w = width + 25
    text = "    -- (distance): [fact]"
    draw.text((text_w, text_h), text, fill="black", font=small_font_bold)
    row += 1

    # draw raking solution
    rake_frames = [image]
    for rake_step in rake_solved.items():
        frame = rake_frames[-1].copy()
        draw = ImageDraw.Draw(frame)
        x, y = rake_step[0]
        rect_start_pos = rect_pos[y][x][0]
        order_num = rake_step[1]
        hue = all_hue_values - (order_num * color_step)
        color = "hsl(%d, %d%%, %d%%)" % (hue, saturation, luminance)

        draw.rectangle(rect_pos[y][x], fill=color, outline="black")
        draw.text(rect_start_pos, str(order_num), fill="black", font=font)
        rake_frames.append(frame)

    if skip_rake:
        frames[-1] = rake_frames[-1]
    else:
        frames.extend(rake_frames)

    # draw properties
    getCC = partial(getCenterCircle, rect_pos, step_half_size, circle_radius)
    draw.ellipse(getCC(properties["start"]), fill="white", outline="white")
    draw.ellipse(getCC(properties["base"]), fill="black", outline="black")
    for point in properties["points"]:
        draw.ellipse(getCC(point), fill="blue", outline="blue")
    frames[-1] = frame  # add these properties into previous frame

    # draw path solution
    x2, y2 = None, None
    fact_iterator = iter(rule_solved.items())
    fact_found = None
    deductions = []  # type: List[str]
    total_dist = 0
    frame = frames[-1].copy()
    saved_frames = [frame]

    for i, path_solved in enumerate(paths_solved, 1):
        # remember last position from previous path
        if x2 is not None:
            path_solved.insert(0, (x2, y2))
        else:
            path_solved.insert(0, properties["start"])

        point_dist = 0
        for j, next_step in enumerate(path_solved[1:]):
            saving_frame = saved_frames[-1].copy()
            draw = ImageDraw.Draw(saving_frame)

            # draw thin lines that will be saved
            x1, y1 = path_solved[j]
            x2, y2 = next_step
            center1 = tuple((c + step_half_size for c in rect_pos[y1][x1][0]))
            center2 = tuple((c + step_half_size for c in rect_pos[y2][x2][0]))
            draw.line((center1, center2), fill="white", width=4)
            saved_frames.append(saving_frame)

            # draw circle and last movement
            showing_frame = saving_frame.copy()
            draw_head = ImageDraw.Draw(showing_frame)
            draw_head.line((center1, center2), fill="black", width=10)
            draw_head.ellipse(getCC(next_step), fill="black", outline="black")
            prev_terr = int(terrained_map[x1][y1])
            next_terr = int(terrained_map[x2][y2])

            # draw distance
            next_dist = path.getNextDist(prev_terr, next_terr, climb)
            point_dist += next_dist
            total_dist += next_dist
            text_w = width + 25
            text_h = height - 25
            text = f"Total distance: {total_dist}"
            draw_head.text(
                (text_w, text_h), text, fill="black", font=small_font_bold
            )
            frames.append(showing_frame)

        # draw point ordering, distances, facts
        if i > 1:  # we have to skip BASE
            fact_found, deductions = next(fact_iterator)
        text_h = row * 15
        text_w = width + 25
        text = f"{i}. -- ({point_dist}): [{fact_found}]"
        draw.text((text_w, text_h), text, fill="black", font=small_font)
        row += 1

        # draw deductions from found fact
        if deductions:
            text_w = width + 50
        for deduction in deductions:
            text_h = row * 15
            text = f"Deduction: {deduction}"
            draw.text((text_w, text_h), text, fill="black", font=small_font)
            row += 1

        # draw path order in map
        center2 = center2[0] + 10, center2[1]
        draw.text(center2, str(i), fill="white", font=font)

    # need to draw again cuz last frame is without total distance
    text_w = width + 25
    text_h = height - 25
    text = f"Total distance: {total_dist}"
    draw.text((text_w, text_h), text, fill="black", font=small_font_bold)
    frames.append(saving_frame)

    # leave the last frame for longer
    frames.extend([frames[-1]] * 20)

    frames[0].save(
        "test.gif",
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=200,
        loop=0,
    )


if __name__ == "__main__":

    fname = "queried"
    skip_rake = True
    climb = True

    createGif(fname, skip_rake, climb)
