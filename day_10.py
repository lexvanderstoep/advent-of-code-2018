import math
import re
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.std import tqdm


@dataclass(frozen=True)
class Vector:
    x: int
    y: int

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)


    def distance(self, other) -> float:
        return math.sqrt(math.pow(self.x - other.x, 2) + math.pow(self.y - other.y, 2))


@dataclass(frozen=True)
class Point:
    position: Vector
    velocity: Vector


def parse_point(file_line: str) -> Point:
    pattern = r"position=<\s*(-?\d+),\s*(-?\d+)> velocity=<\s*(-?\d+),\s*(-?\d+)>"
    match = re.match(pattern, file_line)

    if not match:
        raise ValueError(f'Unable to parse input line {file_line}, expected pattern {pattern}')

    pos_x, pos_y, vel_x, vel_y = map(int, match.groups())
    return Point(Vector(pos_x, pos_y), Vector(vel_x, vel_y))


def parse_input(file_path: str) -> list[Point]:
    with open(file_path, 'r') as file:
        return [parse_point(line) for line in file.readlines()]


def step(points: list[Point]) -> list[Point]:
    return [Point(p.position + p.velocity, p.velocity) for p in points]


def area_score(points: list[Point]) -> int:
    min_x = min([p.position.x for p in points])
    max_x = max([p.position.x for p in points])
    min_y = min([p.position.y for p in points])
    max_y = max([p.position.y for p in points])
    return (max_x - min_x) * (max_y - min_y)


def unique_lines_score(points: list[Point]) -> int:
    unique_x = set([p.position.x for p in points])
    unique_y = set([p.position.y for p in points])
    return len(unique_x) + len(unique_y)


def standard_deviation_score(points: list[Point]) -> float:
    median_point = Vector(
        int(sum([p.position.x for p in points]) / len(points)),
        int(sum([p.position.y for p in points]) / len(points))
    )

    std_deviation = sum([math.pow(p.position.distance(median_point), 2) for p in points]) / len(points)
    return std_deviation


def normalise_scores(scores: pd.Series) -> pd.Series:
    if scores.min() == scores.max():
        return scores / scores.max() - 0.5
    return (scores - scores.min()) / (scores.max() - scores.min())


def plot_grid(points: list[Point]):
    x_coords = [p.position.x for p in points]
    y_coords = [p.position.y for p in points]

    plt.scatter(x_coords, y_coords, color='blue')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.xlim(min(x_coords) - 2, max(x_coords) + 2)
    plt.ylim(min(y_coords) - 2, max(y_coords) + 2)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def compute_part_one(file_name: str):
    points_timelines = [parse_input(file_name)]

    scores = pd.DataFrame(columns=['area', 'unique_lines', 'std_dev'])

    for _ in tqdm(range(20000)):
        points = points_timelines[-1]
        scores.loc[len(scores)] = {
            'area': math.log(area_score(points)),
            'unique_lines': math.log(unique_lines_score(points)),
            'std_dev': math.log(standard_deviation_score(points)),
        }
        points_timelines.append(step(points))

    scores = scores.apply(normalise_scores, axis='rows')
    minimum_scores = scores.idxmin()
    mode_minimum_index = minimum_scores.mode().iloc[0]

    scores.plot(label=minimum_scores)
    plt.show()
    plot_grid(points_timelines[mode_minimum_index])

    print(minimum_scores)


if __name__ == '__main__':
    compute_part_one('input/day_10.txt')
