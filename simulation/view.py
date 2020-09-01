import json
import pickle
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

import model.evolution as evo
import model.pathfinding as path


def loadPickle(fname: str):
    """Loads a pickle file.

    Args:
        fname (str): root name of the file to load

    Returns:
        Any: pickled content
    """

    with open("simulation/solutions/" + fname, "rb") as handle:
        return pickle.loads(handle.read())


def loadJson(fname: str) -> Dict[str, Any]:
    """Loads a pickle file.

    Args:
        fname (str): root name of the file to load

    Returns:
        Dict[str, Any]: json-ed content
    """

    with open("simulation/solutions/" + fname + ".json", "r") as f:
        return json.load(f)


def createGif(fname: str, skip_rake: bool, climb: bool) -> None:
    """Creates gif animation that visualizes the solution.

    Args:
        fname (str): root name of the file to load
        skip_rake (bool): skips the raking part
        climb (bool): Climbing distance approach. If True, distance is measured
            with abs(current terrain number - next terrain number)
    """

    step_size = 50

    # NOTE: visualization only works on rake_solved maps
    try:
        map_ = path.Map(fname)
        height = map_.height * step_size
        width = map_.width * step_size
        properties = map_.properties

        terrained_map = evo.loadMap(fname + "_ter")
        rake_solved = loadPickle(fname + "_rake")
        path_solved = loadPickle(fname + "_path")
        rule_solved = loadJson(fname + "_rule")
    except FileNotFoundError as e:
        print(e)
        return

    # text and drawing sizes
    step_size = 50
    step_half_size = int(step_size / 2)
    window_width = width + 200
    window_height = height + 1

    # get stepping color
    all_hue_values = 180
    last_value = tuple(rake_solved.values())[-1]
    color_step = int(all_hue_values / last_value)

    # create first image and font
    image = Image.new(
        mode="RGB", size=(window_width, window_height), color="white"
    )
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial", step_half_size)
    frames = [image]

    # create window of empty rectangles and unpassable locations
    # also save rectangle coordinates
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

    # create raking solution
    saturation, luminance = 100, 50
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
        frames.append(rake_frames[-1])
    else:
        frames.extend(rake_frames)

    # create properties
    circle_radius = int(step_size / 5)

    

    x2, y2 = None, None
    frame = frames[-1].copy()
    saving_frames = [frame]
    total_dist = 0
    point_dists = []
    for path_s in path_solved:

        # remember last position from previous path
        if x2 is not None:
            path_s.insert(0, (x2, y2))

        point_dist = 0
        for i, next_step in enumerate(path_s[1:]):
            saving_frame = saving_frames[-1].copy()
            draw = ImageDraw.Draw(saving_frame)

            x1, y1 = path_s[i]
            x2, y2 = next_step
            center1 = tuple((c + step_half_size for c in rect_pos[y1][x1][0]))
            center2 = tuple((c + step_half_size for c in rect_pos[y2][x2][0]))

            draw.line((center1, center2), fill="white", width=4)
            # draw properties

            saving_frames.append(saving_frame)

            # draw circle and last movement
            showing_frame = saving_frame.copy()
            draw_head = ImageDraw.Draw(showing_frame)
            draw_head.line((center1, center2), fill="black", width=10)
            draw_head.ellipse(
                (
                    center2[0] - circle_radius,
                    center2[1] - circle_radius,
                    center2[0] + circle_radius,
                    center2[1] + circle_radius,
                ),
                fill="black",
                outline="black",
            )
            prev_terr = int(terrained_map[x1][y1])
            next_terr = int(terrained_map[x2][y2])

            # draw distance
            next_dist = path.getNextDist(prev_terr, next_terr, climb)
            point_dist += next_dist
            total_dist += next_dist
            text_w = width + 25
            text_h = height - 50
            draw_head.text(
                (text_w, text_h), str(total_dist), fill="black", font=font
            )

            frames.append(showing_frame)

        point_dists.append(point_dist)

    # TODO: add rules (each rule add distance)

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
