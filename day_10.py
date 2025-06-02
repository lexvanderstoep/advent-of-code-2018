import math
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.std import tqdm


@dataclass(frozen=True)
class Grid:
    positions: np.array
    velocities: np.array


def parse_point(file_line: str) -> ((int, int), (int, int)):
    pattern = r"position=<\s*(-?\d+),\s*(-?\d+)> velocity=<\s*(-?\d+),\s*(-?\d+)>"
    match = re.match(pattern, file_line)

    if not match:
        raise ValueError(f'Unable to parse input line {file_line}, expected pattern {pattern}')

    pos_x, pos_y, vel_x, vel_y = map(int, match.groups())
    return (pos_x, pos_y), (vel_x, vel_y)


def points_to_grid(points: list[tuple[tuple[int, int], tuple[int, int]]]) -> Grid:
    positions: np.array = np.array([(p[0][0], p[0][1]) for p in points])
    velocities: np.array = np.array([(p[1][0], p[1][1]) for p in points])
    return Grid(positions, velocities)


def parse_input(file_path: str) -> Grid:
    with open(file_path, 'r') as file:
        return points_to_grid([parse_point(line) for line in file.readlines()])


def step(grid: Grid) -> Grid:
    return Grid(grid.positions + grid.velocities, grid.velocities)


def cross_section_score(grid: Grid) -> float:
    min_x = min(grid.positions[:, 0])
    max_x = max(grid.positions[:, 0])
    min_y = min(grid.positions[:, 1])
    max_y = max(grid.positions[:, 1])
    return math.sqrt((max_x - min_x) * (max_y - min_y))


def unique_lines_score(grid: Grid) -> int:
    unique_x = np.unique(grid.positions[:, 0])
    unique_y = np.unique(grid.positions[:, 1])
    return len(unique_x) + len(unique_y)


def standard_deviation_score(grid: Grid) -> float:
    median_point = np.mean(grid.positions, axis=0)
    distances = np.linalg.norm(grid.positions - median_point, axis=1)
    return np.std(distances)


def normalise_scores(scores: pd.Series) -> pd.Series:
    if scores.min() == scores.max():
        return scores / scores.max() - 0.5
    return (scores - scores.min()) / (scores.max() - scores.min())


def plot_grid(grid: Grid):
    x_coords = grid.positions[:, 0]
    y_coords = grid.positions[:, 1]

    plt.scatter(x_coords, y_coords, color='blue')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.xlim(min(x_coords) - 2, max(x_coords) + 2)
    plt.ylim(min(y_coords) - 2, max(y_coords) + 2)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def compute_part_one(file_name: str):
    grid_timeline = [parse_input(file_name)]

    scores = pd.DataFrame(columns=['cross_section', 'unique_lines', 'std_dev'])


    for _ in tqdm(range(20000)):
        grid = grid_timeline[-1]
        scores.loc[len(scores)] = {
            'cross_section': cross_section_score(grid),
            'unique_lines': unique_lines_score(grid),
            'std_dev': standard_deviation_score(grid),
        }
        grid_timeline.append(step(grid))

    scores = scores.apply(normalise_scores, axis='rows')
    minimum_scores = scores.idxmin()
    mode_minimum_index = minimum_scores.mode().iloc[0]

    scores.plot(label=minimum_scores)
    plt.show()
    plot_grid(grid_timeline[mode_minimum_index])

    print(minimum_scores)


if __name__ == '__main__':
    compute_part_one('input/day_10.txt')
