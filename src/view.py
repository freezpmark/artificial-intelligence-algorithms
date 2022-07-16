"""This module serves to create a visualizing simulation of all used stages.

It creates a gif file that shows the outcome of all the AI algorithms.
(stage_1_evolution.py, stage_2_pathfinding.py, stage_3_forward_chain.py).

Function hierarchy:
create_gif
    _load_pickle        - loads solutions (rakes, paths)
    _load_json          - loads solution (facts)
    _save_gif           - creates gif from given frames
    _get_center_circle  - gets the center coordinate of a point
"""

import json
import pickle
from functools import partial
from typing import Any, Dict, List, Tuple
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import stage_1_evolution
import stage_2_pathfinding


def _load_pickle(fname: str, suffix: str) -> Any:
    """Loads a pickle file that has the solution. (rakes, paths)

    The solution is being loaded from /data/solutions directory.

    Args:
        fname (str): name of the file to load
        suffix (str): suffix of fname

    Returns:
        Any: pickled content
    """

    source_dir = Path(__file__).parents[0]
    fname_path = Path(f"{source_dir}/data/solutions/{fname}{suffix}")
    with open(fname_path, "rb") as handle:
        return pickle.loads(handle.read())


def _load_json(fname: str, suffix: str) -> Dict[str, Any]:
    """Loads a json file that has the solution. (facts)

    The solution is being loaded from /data/solutions directory.

    Args:
        fname (str): name of the file to load
        suffix (str): suffix of fname

    Returns:
        Dict[str, Any]: json-ed content
    """

    source_dir = Path(__file__).parents[0]
    fname_path = Path(f"{source_dir}/data/solutions/{fname}{suffix}.json")
    with open(fname_path, encoding="utf-8") as file:
        return json.load(file)


def _save_gif(fname: str, frames: List[Any]) -> None:
    """Saves all frames into gif file.

    Saves the gif animation into /data directory.

    Args:
        fname (str): name of the file that is going to be created
        frames (Image]): drawn images
    """

    source_dir = Path(__file__).parents[0]
    fname_path = Path(f"{source_dir}/data/{fname}.gif")
    frames[0].save(
        fname_path,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=250,
        loop=0,
    )


def _get_center_circle(
    rect_pos: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]],
    step_half_size: int,
    circle_radius: int,
    point_coordinate: Tuple[int, int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Gets coordinates for drawing circle at the center of rectangle.

    Args:
        rect_pos (List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]):
            rectangle coordinate system
        step_half_size (int): half size of a rectangle step
        circle_radius (int): size of circle
        point_coordinate (Tuple[int, int]): coordinate of a point that is used
            as an index for rect_pos

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: box coordinates of circle
    """

    i, j = point_coordinate
    center = tuple((c + step_half_size for c in rect_pos[j][i][0]))
    square_edge1 = tuple((start - circle_radius for start in center))
    square_edge2 = tuple((end + circle_radius for end in center))

    return square_edge1, square_edge2


def create_gif(fname: str, skip_rake: bool, climb: bool) -> None:
    """Creates gif animation that visualizes the solution.

    Args:
        fname (str): name of the file to load
        skip_rake (bool): skips the raking part
        climb (bool): Climbing distance approach. If True, distance is measured
            with abs(current terrain number - next terrain number)
    """

    try:
        map_props = stage_2_pathfinding.Map(fname)
        terrained_map = stage_1_evolution.load_map(fname, "_ter")
        rake_solved = _load_pickle(fname, "_rake")
        paths_solved = _load_pickle(fname, "_path")
        rule_solved = _load_json(fname, "_rule")
    except FileNotFoundError as err:
        print(f"Invalid file name! ({err})")
        exit()

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

    # get raking colors, sat/lum, font
    all_hue_values = 180
    last_rake_value = tuple(rake_solved.values())[-1]
    color_step = int(all_hue_values / last_rake_value)
    saturation = 100
    luminance = 50
    font = ImageFont.truetype("arial", step_half_size)
    small_font = ImageFont.truetype("arial", int(step_half_size / 2))
    small_font_bold = ImageFont.truetype("arialbd", int(step_half_size / 2))

    # create first image
    image = Image.new(mode="RGB", size=(map_width, map_height), color="white")
    draw = ImageDraw.Draw(image)
    frames = [image]

    # draw window of empty rectangles and unpassable locations
    # store rectangled size coordinate system
    rects = []  # type: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]
    for i, x in enumerate(range(0, width, step_size)):
        rects.append([])
        for j, y in enumerate(range(0, height, step_size)):
            rect = (x, y), (x + step_size, y + step_size)
            rects[i].append(rect)
            if terrained_map[j][i] == "-2":
                draw.rectangle(rect, fill=(0, 127, 255), outline="black")
            elif terrained_map[j][i] == "-1":
                draw.rectangle(rect, fill="black", outline="black")
            else:
                draw.rectangle(rect, outline="black")

    # draw facts text explanation
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
        order_num = rake_step[1]
        hue = all_hue_values - (order_num * color_step)
        color = f"hsl({hue}, {saturation}%, {luminance}%)"

        rect_start_pos = rects[y][x][0]
        draw.rectangle(rects[y][x], fill=color, outline="black")
        draw.text(rect_start_pos, str(order_num), fill="black", font=font)
        rake_frames.append(frame)
    if skip_rake:
        frames[-1] = rake_frames[-1]
    else:
        frames.extend(rake_frames)

    # draw properties
    get_map_coordinate = partial(
        _get_center_circle, rects, step_half_size, circle_radius
    )
    start_coor = get_map_coordinate(properties["start"])
    home_coor = get_map_coordinate(properties["home"])
    draw.ellipse(start_coor, fill="white", outline="white")
    draw.ellipse(home_coor, fill="black", outline="black")
    for point in properties["points"]:
        point_coordinate = get_map_coordinate(point)
        draw.ellipse(point_coordinate, fill="blue", outline="blue")

    # draw path solution
    x_head, y_head = None, None
    fact_iterator = iter(rule_solved.items())
    fact_found = None
    deductions = []  # type: List[str]
    total_dist = 0
    frame = frames[-1].copy()
    saved_frames = [frame]

    for i, path_solved in enumerate(paths_solved, 1):
        # remember last position from the previous path
        if x_head is not None:
            path_solved.insert(0, (x_head, y_head))
        else:
            path_solved.insert(0, properties["start"])

        point_dist = 0
        for j, next_step in enumerate(path_solved[1:]):
            saving_frame = saved_frames[-1].copy()
            draw = ImageDraw.Draw(saving_frame)

            # draw thin lines that will persist
            x_tail, y_tail = path_solved[j]
            x_head, y_head = next_step
            rect_tail = rects[y_tail][x_tail][0]
            rect_head = rects[y_head][x_head][0]
            center_tail = tuple((c + step_half_size for c in rect_tail))
            center_head = tuple((c + step_half_size for c in rect_head))
            draw.line((center_tail, center_head), fill="white", width=4)
            saved_frames.append(saving_frame)

            # draw circle and last movement
            next_step_coor = get_map_coordinate(next_step)
            showing_frame = saving_frame.copy()
            draw_head = ImageDraw.Draw(showing_frame)
            draw_head.line((center_tail, center_head), fill="black", width=10)
            draw_head.ellipse(next_step_coor, fill="black", outline="black")
            prev_terr = int(terrained_map[x_tail][y_tail])
            next_terr = int(terrained_map[x_head][y_head])

            # draw distance
            # need parameter climb to draw distance because we are using
            # get_next_dist function that is in the path module. It would be a
            # hassle to compute the distances and putting them to results
            next_dist = stage_2_pathfinding.get_next_dist(
                prev_terr, next_terr, climb
            )
            point_dist += next_dist
            total_dist += next_dist
            text_w = width + 25
            text_h = height - 25
            text = f"Total distance: {total_dist}"
            draw_head.text(
                (text_w, text_h), text, fill="black", font=small_font_bold
            )
            frames.append(showing_frame)

        # draw visited points, distances and facts
        if i > 1:  # we have to skip HOME
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

        # draw visiting order number of points in the map
        center_head = center_head[0] + 10, center_head[1]
        draw.text(center_head, str(i), fill="white", font=font)

    # draw total distance again for the ending frame that has no tail or head
    text_w = width + 25
    text_h = height - 25
    text = f"Total distance: {total_dist}"
    draw.text((text_w, text_h), text, fill="black", font=small_font_bold)

    # ending frame, make it last longer
    frames.append(saving_frame)
    frames.extend([frames[-1]] * 30)

    _save_gif(fname, frames)


if __name__ == "__main__":

    FNAME = "queried"
    SKIP_RAKE = True
    CLIMB = True

    view_parameters = dict(
        fname=FNAME,
        skip_rake=SKIP_RAKE,
        climb=CLIMB,
    )  # type: Dict[str, Any]

    create_gif(**view_parameters)
